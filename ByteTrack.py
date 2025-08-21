import argparse
from ultralytics_folder.trackers import byte_tracker

class BYTETrack:
    def __init__(self):
        args = argparse.Namespace(
            track_high_thresh=0.3, # Minimum detection confidence for a track to be considered valid (set lower than default to accept more detections)
            track_low_thresh=0.01, # Minimum confidence to keep an existing track alive (very low to avoid losing tracks too quickly)
            track_thresh=0.1,      # Threshold for associating detections with existing tracks (low to allow looser matching)
            track_buffer=150,      # Number of frames to keep a lost track before removing it (high to allow short disappearances)
            match_thresh=0.7,      # IoU threshold to match detections to existing tracks (high to require strong overlap)
            new_track_thresh=0.05, # Minimum confidence to start a new track (low to allow easier creation of new tracks)
            fuse_score=False,      # Whether to combine detection scores with track scores (disabled)
            iou=0.05,              # Minimum IoU required to match a detection to an existing track (very low to allow for plenty of movement)
            proximity_thresh=600,  # Maximum pixel distance to match objects between frames (large to handle fast movement)
            agnostic_nms=True,     # Ignore class labels when removing overlapping boxes
        )

        self.model = byte_tracker.BYTETracker(args)

    def track(self, detections):
        """
        Update the tracker with Ultralytics Results for one frame.

        Parameters
        ----------
        detections: ultralytics.engine.results.Results
            Ultralytics Results object for the current frame.

        Returns
        -------
        np.ndarray
            Array of shape (N, 7), where each row is:
            [x1, y1, x2, y2, class_id, confidence, track_id].
        """

        # Extract arrays in CPU numpy format
        xywh = detections.boxes.xywh.cpu().numpy()   # Box coordinates: center x, center y, width, height
        conf = detections.boxes.conf.cpu().numpy()   # Confidence scores for each detection
        classes = detections.boxes.cls.cpu().numpy() # Class IDs for each detection
        
        results = argparse.Namespace(xywh=xywh, conf=conf, cls=classes)
        return self.model.update(results)
    