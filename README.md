# EasyMediapipe
EasyMediapipe is a Python package that simplifies the use of the Mediapipe library for computer vision applications.

## Installation
To install EasyMedipe, you need to have Python 3.11 or higher installed on your system. You can install EasyMediapipe using pip:

`pip install EasyMediapipe`

<hr>

# Examples

## Pose

```python

from EasyMediapipe.Pose import Pose

pose = Pose(num_poses=1)
print(pose.detect_pose(pose.read_image("IMAGE_PATH")))

```

## Hand

```python

from EasyMediapipe.Hand import Hand

hand = Hand(num_hands=2)
print(hand.detect_hand(hand.read_image("IMAGE_PATH")))

```

## Face Detection 

```python

from EasyMediapipe.FaceDetection import Face


face  = Face()
print(face.detect_face(face.read_image("IMAGE_PATH")))

```

