import torch
import numpy as np
from ultralytics import YOLOWorld

class YoloModel():
    def __init__(self, model_name="yolov8x-worldv2.pt"):
        # In "yolov8x-worldv2.pt", the "x" can be replaced with:
        #   n = nano, s = small, m = medium, l = large, x = extra-large
        self.model = YOLOWorld(model_name)  # Load the YOLO-World model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device) # Use GPU if available, else CPU
        self.device = device

        # Optional warm-up to avoid long delay on first detection
        # Without this, the first inference after model load will be around 20x slower
        dummy = np.zeros((640, 480, 3), dtype=np.uint8)
        self.find_objects(dummy)

    
    def update_classes(self, class_list):
        """Set the list of target classes for open-vocabulary detection."""
        self.model.set_classes(class_list)


    def find_objects(self, frame, verbose=False):
        """Run object detection on a single frame."""
        # Get predictions from the model. show=True gives slow results
        results = self.model.predict(
            source=frame,
            verbose=verbose,   # Whether to print detailed inference logs     
            conf=0.05,         # Low threshold to detect even low-confidence objects
            iou=0.5,           # Intersection over Union threshold for removing duplicate boxes
            augment=True,      # Apply test-time augmentations (slower but may help with motion blur)
            agnostic_nms=True, # If boxes overlap a lot, keep only the one with higher confidence, even if labels differ
            max_det=100        # Limit the number of detections per image
        )
        return results[0] # We only have one image, so just return its result
    