# Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2,wget
from pathlib import Path
from EasyMediapipe.utilities import draw_landmarks_on_pose_image

class Pose():
    def __init__(self,model_path : str="pose_landmarker.task",
                 output_segmentation_masks:bool = False,
                 running_mode : str = "image",
                 num_poses : int = 1,
                 min_pose_detection_confidence: float = 0.5,
                 min_pose_presence_confidence : float = 0.5,
                 min_tracking_confidence : float = 0.5,
                 draw: bool = True
                 ):
        self.running_mode = running_mode
        if self.running_mode == "image":
            self.running_mode = mp.tasks.vision.RunningMode.IMAGE
        else: 
            self.running_mode = mp.tasks.vision.RunningMode.VIDEO
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_pose_presence_confidence = min_pose_presence_confidence
        self.model_asset_path  = model_path
        self.draw = draw
        self.model_path = Path(self.model_asset_path)
        if self.model_path.exists():
            print("Loaded The Model!!")
        else:
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            # Download the file using wget
            try:
                wget.download(url, out=self.model_asset_path)
                print(f"File downloaded successfully to: {self.model_asset_path}")
            except Exception as e:
                print(f"Error downloading file. {e}")

        self.output_segmentation_masks = output_segmentation_masks
        self.base_options = python.BaseOptions(model_asset_path=self.model_asset_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=self.output_segmentation_masks,
            running_mode = self.running_mode,
            min_pose_detection_confidence = self.min_pose_detection_confidence,
            min_pose_presence_confidence = self.min_pose_presence_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
            num_poses = self.num_poses
            )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

    def read_image(self,image_path):
        return cv2.imread(image_path)


    def detect_pose(self, image):
        
        total_points = {}
        h,w  = image.shape[:2]
        image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(image_rgb)
        pose_landmarks_list  = detection_result.pose_landmarks
        if pose_landmarks_list:
            for person_idx in range(len(pose_landmarks_list)):
                points = {}
                landmarks = pose_landmarks_list[person_idx]
                for points_idx,landmark in enumerate(landmarks):
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = int(landmark.z * w)
                    visibility = landmark.visibility
                    points[points_idx]  = [x,y,z,visibility]
                total_points[person_idx] = points
        if self.draw:
            image = draw_landmarks_on_pose_image(image,detection_result)
            
        return total_points,image
        
    




