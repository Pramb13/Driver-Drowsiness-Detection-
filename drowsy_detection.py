import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def get_mediapipe_app(max_num_faces=1):
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=max_num_faces, refine_landmarks=True)


def distance(point_1, point_2):
    return sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = [denormalize_coordinates(landmarks[i].x, landmarks[i].y, frame_width, frame_height) for i in refer_idxs]

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        return (P2_P6 + P3_P5) / (2.0 * P1_P4), coords_points
    except:
        return 0.0, None


class VideoFrameHandler:
    def __init__(self):
        self.eye_idxs = {"left": [362, 385, 387, 263, 373, 380], "right": [33, 160, 158, 133, 153, 144]}
        self.RED, self.GREEN = (0, 0, 255), (0, 255, 0)
        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = {"start_time": time.perf_counter(), "DROWSY_TIME": 0.0, "COLOR": self.GREEN, "play_alarm": False}

    def process(self, frame, thresholds):
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear, left_coords = get_ear(landmarks, self.eye_idxs["left"], frame_w, frame_h)
            right_ear, right_coords = get_ear(landmarks, self.eye_idxs["right"], frame_w, frame_h)
            EAR = (left_ear + right_ear) / 2.0

            if EAR < thresholds["EAR_THRESH"]:
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
            else:
                self.state_tracker.update({"start_time": time.perf_counter(), "DROWSY_TIME": 0.0, "COLOR": self.GREEN, "play_alarm": False})

        return frame, self.state_tracker["play_alarm"]
