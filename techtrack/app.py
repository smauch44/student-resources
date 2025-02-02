import cv2
from modules.inference.nms import NMS
from modules.inference.model import Detector
from modules.inference.preprocessing import Preprocessing


class InferenceService:
    """
    Handles real-time video inference by integrating preprocessing, object detection, 
    and Non-Maximum Suppression (NMS) filtering.

    This service continuously captures video frames, applies object detection, 
    filters results using NMS, and outputs predictions.
    """

    def __init__(self, stream: Preprocessing, detector: Detector, nms: NMS, drop_rate: int = 10) -> None:
        """
        Initializes the inference service.

        :param stream: An instance of the Preprocessing class for video capture.
        :param detector: An instance of the Model class for object detection.
        :param nms: An instance of the NMS class for filtering overlapping bounding boxes.
        :param drop_rate: The rate at which frames are dropped (default: 10).

        :ivar self.stream: Video stream processor.
        :ivar self.detector: Object detection model.
        :ivar self.nms: Non-Maximum Suppression processor.
        """
        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.drop_rate = drop_rate

        print("[INFO] Inference Service initialized.")

    def run(self) -> None:
        """
        Runs the inference service.

        This method:
        1. Captures frames from the video stream.
        2. Applies object detection to each frame.
        3. Filters the detected bounding boxes using Non-Maximum Suppression (NMS).
        4. Prints the results.

        **Processing Pipeline:**
        - Capture video frame.
        - Perform object detection using YOLO.
        - Post-process detections to extract bounding boxes and confidence scores.
        - Apply Non-Maximum Suppression to remove redundant detections.
        - Display the predictions and filtered results.

        **Example Output:**
        ```
        Predictions: [[x, y, w, h], ...] [class_ids] [confidence_scores] [class_scores]
        Filtered: [[x, y, w, h], ...] [class_ids] [confidence_scores] [class_scores]
        ```

        **Example Usage:**
        ```python
        service = InferenceService(stream, model, nms)
        service.run()
        ```
        """
        
        # TASK 5: Implement your Inference service. Use Line 40-51 to guide you on the
        #         logic behind this implementation. 


# Runner for Inference Service
if __name__ == "__main__":
    print("[INFO] Starting Inference Service...")

    # Initialize Video Processing
    VIDEO_SOURCE = "udp://127.0.0.1:23000"
    print(f"[INFO] Initializing video stream from source: {VIDEO_SOURCE}")
    stream = Preprocessing(VIDEO_SOURCE)

    # Initialize Model
    WEIGHTS_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights"
    CONFIG_PATH = "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg"
    CLASS_NAMES_PATH = "storage/yolo_models/logistics.names"

    print("[INFO] Loading YOLO Model...")
    model = Detector(WEIGHTS_PATH, CONFIG_PATH, CLASS_NAMES_PATH)
    print("[INFO] Model loaded successfully.")

    # Initialize NMS
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    print("[INFO] Initializing Non-Maximum Suppression...")
    nms = NMS(SCORE_THRESHOLD, IOU_THRESHOLD)
    print("[INFO] NMS initialized.")

    # Create and run inference service
    print("[INFO] Starting Inference Service loop...")
    service = InferenceService(stream, model, nms)
    service.run()

    print("[INFO] Inference Service terminated.")
