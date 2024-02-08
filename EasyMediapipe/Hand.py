# Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2,wget
from pathlib import Path
from EasyMediapipe.utilities import draw_landmarks_on_hand_image

class Hand():
    def __init__(self,model_path : str="hand_landmarker.task",
                num_hands : int = 2,
                min_hand_detection_confidence: float = 0.5,
                min_hand_presence_confidence : float = 0.5,
                min_tracking_confidence : float = 0.5,
                draw: bool = True
                ):
        self.running_mode = mp.tasks.vision.RunningMode.IMAGE
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.model_asset_path  = model_path
        self.draw = draw
        self.model_path = Path(self.model_asset_path)
        if self.model_path.exists():
            print("Loaded The Model!!")
        else:
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            # Download the file using wget
            try:
                wget.download(url, out=self.model_asset_path)
                print(f"File downloaded successfully to: {self.model_asset_path}")
            except Exception as e:
                print(f"Error downloading file. {e}")

        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode = self.running_mode,
            min_hand_detection_confidence = self.min_hand_detection_confidence,
            min_hand_presence_confidence = self.min_hand_presence_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
            num_hands = self.num_hands
            )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def read_image(self,image_path):
        return cv2.imread(image_path)


    def detect_hand(self, image):
        
        total_points = {}
        h,w  = image.shape[:2]
        image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(image_rgb)
        hand_landmarks_list  = detection_result.hand_landmarks
        if hand_landmarks_list:
            for person_idx in range(len(hand_landmarks_list)):
                points = {}
                landmarks = hand_landmarks_list[person_idx]
                for points_idx,landmark in enumerate(landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = int(landmark.z * w)
                    visibility = landmark.visibility
                    points[points_idx]  = [x,y,z,visibility]
                total_points[person_idx] = points
        if self.draw:
            image = draw_landmarks_on_hand_image(image,detection_result)
            
        return total_points,image
        
    




