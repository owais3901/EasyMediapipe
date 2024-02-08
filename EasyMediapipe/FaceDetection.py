
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import wget
import cv2
from EasyMediapipe.utilities import draw_box_on_detection




class Face():
    def __init__(self,model_path : str="detector.tflite",
                 min_detection_confidence: float = 0.5,
                 min_suppression_threshold : float = 0.3,
                 draw: bool = True
                 ):
        self.running_mode = mp.tasks.vision.RunningMode.IMAGE

        self.min_detection_confidence = min_detection_confidence
        self.min_suppression_threshold = min_suppression_threshold
        self.model_asset_path  = model_path
        self.draw = draw
        self.model_path = Path(self.model_asset_path)
        if self.model_path.exists():
            print("Loaded The Model!!")
        else:
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            # Download the file using wget
            try:
                wget.download(url, out=self.model_asset_path)
                print(f"File downloaded successfully to: {self.model_asset_path}")
            except Exception as e:
                print(f"Error downloading file. {e}")

        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.FaceDetectorOptions(
            base_options=self.base_options,
            running_mode = self.running_mode,
            min_detection_confidence = self.min_detection_confidence,
            min_suppression_threshold = self.min_suppression_threshold
            )
        self.detector = vision.FaceDetector.create_from_options(self.options)
    
    def read_image(self,image_path):
        return cv2.imread(image_path)


    def detect_face(self, image):
        total_points = {}
        image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(image_rgb)
        face_list  = detection_result.detections
        if face_list:
            for person_idx in range(len(face_list)):
                bbox = face_list[person_idx].bounding_box
                total_points[person_idx] = [bbox.origin_x,bbox.origin_y,bbox.width,bbox.height]
        if self.draw:
            image = draw_box_on_detection(image,detection_result)
           
        return total_points,image
        
    




