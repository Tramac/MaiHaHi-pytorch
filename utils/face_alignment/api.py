import sys
import cv2
import numpy as np

from .utils import *
from enum import Enum


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    def __init__(self,
                 landmarks_type,
                 network_size=NetworkSize.LARGE,
                 device='cuda',
                 flip_input=False,
                 face_detector='sfd',
                 facedetectmodelfile="./checkpoints/s3fd-619a316812.pth",
                 verbose=False):
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        self.device = device

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__(
            'utils.face_alignment.detection.' + face_detector,
            globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(
            device=device, path_to_detector=facedetectmodelfile, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results

    def get_detections_for_image(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces[0]):
            if len(d) == 0:
                results.append(None)
                continue
            d = np.clip(d, 0, None)

            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results
