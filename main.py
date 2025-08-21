# Standard library
import argparse
import ast
import csv
import os
from argparse import BooleanOptionalAction
from math import floor
from time import time

# Third-party
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Barrier, Manager, Queue

# Local application
import ByteTrack
import GPT
import Yolo
from ultralytics_folder.engine.results import Results as UltralyticsResults



def yolo_run(shared_data, setup_barrier, raw_frame_q, detection_result_q, detectable_classes_q, num, verbose):
    """Runs a YOLO object detection process that alternates between two modes:
        1. Active mode: reads frames, performs object detection, replaces detailed labels with short labels,
        and sends the results to bytetrack_run.
        2. Inactive mode: waits for updated class lists from lmm_run and updates the YOLO model.
    """
    yolo = Yolo.YoloModel()
    name = f"yolo{num}"   # Unique process name, either "yolo1" or "yolo2"
    class_dict = {}       # A dict like {"yellow cornstarch box": "cornstarch"}
    short_labels_str = "" # A string like "apples, cornstarch, oranges, cutting board"

    print(f"setup_barrier reached in yolo_run ({name})")
    setup_barrier.wait()  # Wait for all processes to finish setup before proceeding

    while True:
        # YOLO process active: processing frames
        if shared_data["active_process"] == name:

            frame = raw_frame_q.get() # Get next raw frame from run()
            if frame is None:
                break

            yolo_result = yolo.find_objects(frame, verbose)

            # Replace YOLO's detailed labels with shorter, more readable ones
            # Example: {0: "apple", 1: "cornstarch"} instead of {0: "red apple", 1: "yellow cornstarch box"}
            yolo_result.names = {
                cls_id: class_dict.get(detailed_label, detailed_label) # Try to map detailed_label to a short label. If missing, leave detailed_label unchanged
                for cls_id, detailed_label in yolo_result.names.items()
            }

            # Send YOLO results and short labels to bytetrack_run
            detection_result_q.put((yolo_result, short_labels_str))

            # Measure frame processing time
            elapsed_time = time() - shared_data["last_frame_time"]
            print(f"{name}: {round(elapsed_time, 5)} s ({floor(1/elapsed_time)} fps)")
            shared_data["frame_processing_times"].append(elapsed_time)
            shared_data["last_frame_time"] = time()

        # YOLO process inactive: wait for class_dict from LMM, then update YOLO
        else:
            class_dict = detectable_classes_q.get() # Get latest class_dict from lmm_run(). Determines which classes YOLO can detect
            if class_dict is None:
                break
            detectable_detailed_labels = list(class_dict.keys()) # detectable_detailed_labels is a list like ["red apple", "yellow cornstarch box", "wooden cutting board"]
            start_time = time()
            yolo.update_classes(detectable_detailed_labels)
            print(f"--------------------------- Ran update_classes in {name} in {time()-start_time} seconds ---------------------------\n...setting detectable_detailed_labels = {detectable_detailed_labels}")
            short_labels_str = ", ".join(sorted(set(class_dict.values()))) # A string like "apples, cornstarch, cutting board"
            shared_data["active_process"] = name # Switch state: this process becomes active
            
    raw_frame_q.put(None)                # Signal termination to other yolo_run process
    detection_result_q.put((None, None)) # Signal termination to bytetrack_run

    # Clean up resources
    for q in [raw_frame_q, detection_result_q]:
        q.close()
        q.cancel_join_thread()



def bytetrack_run(setup_barrier, detection_result_q, tracking_results_q):
    """Tracks detections across frames using ByteTrack. Outputs tracking results.

    Workflow:
        1. Receives YOLO detection results (Ultralytics Results) from detection_result_q.
        2. Runs ByteTrack to assign track IDs across frames.
        3. Wraps the tracking output in an Ultralytics Results object.
        4. Sends tracking results to record_and_display_results via tracking_results_q.
    """
    bytetrack = ByteTrack.BYTETrack()
    print("setup_barrier reached in bytetrack_run")
    setup_barrier.wait()  # Wait for all processes to finish setup before proceeding
    
    while True:
        yolo_result, short_labels_str = detection_result_q.get() # Get YOLO detection results from yolo_run()
        if yolo_result is None:
            break
        tracks = bytetrack.track(yolo_result) # ndarray of shape (N, 7): [x1, y1, x2, y2, class_id, confidence, track_id]
        wrapped_results = wrap_tracks_as_results(tracks, yolo_result) # Create an Ultralytics Results object that combines YOLO metadata with ByteTrack IDs

        # Send tracking results and short labels to record_and_display_results
        tracking_results_q.put((wrapped_results, short_labels_str))

    tracking_results_q.put((None, None)) # Signal termination to record_and_display_results
    print("Returning from bytetrack_run")



def wrap_tracks_as_results(tracks, yolo_result):
    """Wraps ByteTrack output in an Ultralytics Results object (same type as YOLO returns).

    Expected tracks layout: (N, 7) float32
        [x1, y1, x2, y2, class_id, confidence, track_id]
    """
    # Empty case: no tracks for this frame
    if tracks is None or len(tracks) == 0:
        tracks = torch.empty((0, 7), dtype=torch.float32)

    # Build a new Results object: keep metadata, replace boxes with tracked outputs
    wrapped_results = UltralyticsResults(
        orig_img=yolo_result.orig_img, # Original image (NumPy array)
        path=yolo_result.path,         # Image path (string) or empty
        names=yolo_result.names,       # Copy mapping, e.g. {0: "apple", 1: "banana"}
        boxes=tracks                   # ndarray of shape (N, 7): [x1, y1, x2, y2, class_id, confidence, track_id]
    )

    return wrapped_results



def get_fourcc_from_path(path):
    """Returns the appropriate FOURCC (Four Character Code) based on file extension."""

    ext = os.path.splitext(path)[1].lower() # Extract file extension (e.g. ".mp4")

    # Mapping from extension to codec string (must be 4 characters)
    ext_to_fourcc = {".mp4": "mp4v", ".avi": "XVID", ".mkv": "X264",}

    try:
        return cv2.VideoWriter_fourcc(*ext_to_fourcc[ext]) # Unpack the 4-char string into individual chars, e.g. "mp4v" → 'm','p','4','v'
    except KeyError:
        raise ValueError(f'Unsupported video extension: {ext}. Supported: {", ".join(ext_to_fourcc)}')


def record_and_display_results(setup_barrier, tracking_results_q, realtime_display, fps, width, height, output_path="output.mp4"):
    """Writes annotated frames to a video file and, optionally, displays them in real time.
    
    Workflow:
        1. Receives tracking results from bytetrack_run via tracking_results_q.
        2. Overlays onto each frame the object classes that are currently detectable.
        3. Records all annotated frames to an output video file using the appropriate codec.
        4. Optionally shows the frames in a resizable OpenCV window with "q" to quit.
    """
    # Layout: margins (px) and text style
    line_char_limit = 120
    left_margin = 10
    bottom_margin = 10
    font_face = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = height / 1080 # Scale text with frame height. Baseline 1080p = 1.0 font_scale
    thickness = 2 # Text thickness

    # Derive line height from font metrics
    (_, ascent_px), descent_px = cv2.getTextSize("Ag", font_face, font_scale, thickness) # Size of "Ag" at this font: ((width, ascent), descent)
    line_height = ascent_px + descent_px + 4 # Text above baseline + below baseline + small padding

    # Set up video writer (codec/fps/size) and window title
    fourcc = get_fourcc_from_path(output_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    window_name = "YOLO-World Detections with Tracking"

    # Create a resizable window only if showing in real time
    if realtime_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    setup_barrier.wait() # Wait for all processes to finish setup before proceeding
    print("setup_barrier reached in record_and_display_results")

    while True:
        tracking_results, short_labels_str = tracking_results_q.get() # Get tracking results from bytetrack_run()
        if tracking_results is None:
            break
        plotted_frame = tracking_results.plot(line_width=3, font_size=3)

        # Wrap text (detectable object classes) into lines
        lines = []
        text = short_labels_str
        while len(text) > line_char_limit:
            i = text.rfind(" ", 0, line_char_limit) # Find the last space before the limit to keep words intact
            if i == -1:  # No space within the limit: force a hard break at the limit
                lines.append(text[:line_char_limit])
                text = text[line_char_limit:]
            else: # Break at that space and exclude it from the next line
                lines.append(text[:i])
                text = text[i+1:]
        lines.append(text) # Append any remaining text

        # Draw text (detectable object classes) bottom up onto plotted_frame (in-place)
        for idx, line in enumerate(reversed(lines)):
            y = height - bottom_margin - descent_px - idx * line_height
            cv2.putText(
                img=plotted_frame,    # Target image (modified in place)
                text=line,            # Text to draw
                org=(left_margin, y), # Bottom-left baseline position (x, y)
                fontFace=font_face,   # OpenCV font family (e.g., FONT_HERSHEY_SIMPLEX)
                fontScale=font_scale, # Relative text size multiplier
                color=(0, 0, 0),      # Text color in BGR (black)
                thickness=thickness,  # Stroke thickness in pixels
                lineType=cv2.LINE_AA  # Anti-aliased line type (smooth edges)
            )

        # Write annotated frame to output video file
        out.write(plotted_frame)  

        # Optionally, display a live preview. Exit on "q"
        if realtime_display:
            cv2.imshow(window_name, plotted_frame) # Show the current annotated frame in a window
            key = cv2.waitKey(1) # Wait 1 ms so the window can refresh and also capture key presses
            if (key & 0xFF) == ord("q"): # Quit display if "q" key is pressed
                cv2.destroyAllWindows()
                realtime_display = False 
        
    cv2.destroyAllWindows() # Also works when no windows are open
    out.release()
    print("Returning from record_and_display_results")



def lmm_run(shared_data, setup_barrier, detectable_classes_q, lmm_raw_frame_q):
    """Runs the LMM to identify object categories that can be detected by YOLO.
    
    Workflow:
        1. Reads a frame from lmm_raw_frame_q sent by run().
        2. Sends the frame to the LMM with the text prompt stored in "prompt.txt".
        3. Parses the LMM response for a dictionary of detectable classes.
        4. Sends this dictionary to yolo_run via detectable_classes_q.
    """
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt_text = f.read()
    lmm = GPT.GPT(prompt_text)
    
    old_class_dict = {}

    print("setup_barrier reached in lmm_run")
    setup_barrier.wait()  # Wait for all processes to finish setup before proceeding
    
    while True:
        # Signaling run() to put a new frame in lmm_raw_frame_q. This design makes sure that lmm_run always
        # has access to the most up-to-date frame at the start of each prompt cycle.
        shared_data["update_lmm_frame_queue"] = True 
        frame = lmm_raw_frame_q.get() # Get latest raw frame from run() for prompting the LMM
        if frame is None:
            break

        # Get GPT's response for objects detected in the frame (expected to include a Python dictionary in the text)
        start_time = time()
        raw_response  = lmm.send_prompt(frame)
        
        # Locate the dictionary portion in the GPT output
        left = raw_response.find("{")       # Find the index of the first "{" in the string (start of dictionary)
        right = raw_response.rfind("}") + 1 # Find the index after the last "}" in the string (end of dictionary, inclusive)
        if left == -1 or right == 0:        # If either "{" or "}" was not found
            raise ValueError("No dictionary found in raw_response in lmm_run")
        class_dict = ast.literal_eval(raw_response[left:right]) # Parsing the substring as a Python dictionary

        print(f"--------------------------- Ran LMM in {time()-start_time} seconds ---------------------------")
        print(f"class_dict: {class_dict}")
        
        # Non-blocking, latest-value-wins design so lmm_run never blocks and yolo_run always receives the most
        # up-to-date class_dict. This design is beneficial if lmm_run is faster than yolo.update_classes
        if class_dict != old_class_dict:
            try:
                detectable_classes_q.put(class_dict, block=False) # Send updated class_dict to yolo_run
            except:
                detectable_classes_q.get()
                detectable_classes_q.put(class_dict) # Send updated class_dict to yolo_run
            old_class_dict = class_dict

    detectable_classes_q.put(None) # Signal termination to inactive yolo_run process
    print("Returning from lmm_run")



def run(input_path, output_path, realtime_display, verbose):
    """Main function that initializes all processes and resources."""
    with Manager() as manager:

        capture = cv2.VideoCapture(input_path)  # Open video file
        if not capture.isOpened():
            print(f'Error: Could not open "{input_path}".')
            return
        
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0: # Some video files/cameras don't store FPS correctly. Default to 30.0 if value is invalid
            fps = 30.0

        shared_data = manager.dict() # Shared dictionary accessible by all processes for synchronization and logging
        shared_data["active_process"] = "yolo1" # Start with yolo1 as the active process
        shared_data["update_lmm_frame_queue"] = False # Used for synchronization so that LMM always gets the latest frame
        shared_data["last_frame_time"] = time()  # Used for logging only
        shared_data["frame_processing_times"] = [] # List of per-frame processing times
        shared_data["classes"] = {} # Used to map detailed description to short labels. Example: {“yellow cornstarch box”: “cornstarch”}. # Updated in lmm_run in the main code.

        setup_barrier = Barrier(5) # Ensures all processes are fully initiated before proceeding
        
        raw_frame_q = Queue(maxsize=1)          # Queue for sending raw frames from run to yolo_run
        lmm_raw_frame_q = Queue(maxsize=1)      # Queue for sending raw frames from run to lmm_run
        detection_result_q = Queue(maxsize=1)   # Queue for sending detection results from yolo_run to bytetrack_run
        tracking_results_q = Queue(maxsize=1)   # Queue for sending tracking results from bytetrack_run to record_and_display_results
        detectable_classes_q = Queue(maxsize=1) # Queue for sending dictionary of detectable classes from lmm_run to yolo_run

        # Create args for both YOLO processes. Avoids unnecessary code repetition
        yolo_args_list = [
            (shared_data, setup_barrier, raw_frame_q, detection_result_q, detectable_classes_q, num, verbose)
            for num in [1, 2]
        ]

        # Create subprocesses
        yolo1_process = mp.Process(target=yolo_run, args=yolo_args_list[0])
        yolo2_process = mp.Process(target=yolo_run, args=yolo_args_list[1])
        bytetrack_process = mp.Process(target=bytetrack_run, args=(setup_barrier, detection_result_q, tracking_results_q))
        display_process = mp.Process(target=record_and_display_results, args=(setup_barrier, tracking_results_q, realtime_display, fps, width, height, output_path))
        lmm_process = mp.Process(target=lmm_run, args=(shared_data, setup_barrier, detectable_classes_q, lmm_raw_frame_q))
        
        # Start subprocesses
        yolo1_process.start()
        yolo2_process.start()
        bytetrack_process.start()
        display_process.start()
        lmm_process.start()

        # Read video and feed frames to YOLO and LMM
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            raw_frame_q.put(frame)  # Send raw frame to yolo_run (every frame)
            if shared_data["update_lmm_frame_queue"]:
                shared_data["update_lmm_frame_queue"] = False
                lmm_raw_frame_q.put(frame) # Send raw frame to lmm_run (occasionally, when lmm_run is ready)

        raw_frame_q.put(None)    # Signaling yolo_run to terminate
        lmm_raw_frame_q.put(None) # Signaling lmm_run to terminate
        capture.release()

        # Wait for processes to finish
        yolo1_process.join()
        print("yolo1_process.join()")
        yolo2_process.join()
        print("yolo2_process.join()")
        bytetrack_process.join()
        print("bytetrack_process.join()")
        display_process.join()
        print("display_process.join()")
        lmm_process.join()
        print("lmm_process.join()\n")

        print("Frame processing times")
        print(f'Average time: {sum(shared_data["frame_processing_times"])/len(shared_data["frame_processing_times"])}')
        print(f'Min time: {min(shared_data["frame_processing_times"])}')
        print(f'Max time: {max(shared_data["frame_processing_times"])}')
        
        # Optionally, write processing time to file
        with open("frame_times.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for t in shared_data["frame_processing_times"]:
                writer.writerow([t])



if __name__ == "__main__":
    # Must be "spawn" to ensure yolo instances are separate. Can be checked by printing id(yolo)
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="DOVRT: Dynamic, Open-Vocabulary, Real-Time Object Detection and Tracking.\nUses YOLO + ByteTrack + GPT")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", nargs="?", default="output.mp4", help="Output video path")
    parser.add_argument("--realtime", action=BooleanOptionalAction, default=True, help="Show live preview (default: on)")
    parser.add_argument("--verbose", action=BooleanOptionalAction, default=False, help="Verbose logging (default: off)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Error: The file "{args.input}" does not exist.')
    else:
        run(args.input, args.output, realtime_display=args.realtime, verbose=args.verbose)
