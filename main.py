from EasyMediapipe.Pose import Pose
from EasyMediapipe.Hand import Hand
from EasyMediapipe.FaceDetection import Face


pose = Pose(num_poses=1)
print(pose.detect_pose(pose.read_image("IMAGE_PATH")))

hand = Hand(num_hands=2)
print(hand.detect_hand(hand.read_image("IMAGE_PATH")))

face  = Face()
print(face.detect_face(face.read_image("IMAGE_PATH")))