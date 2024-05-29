import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_body_angles(landmarks):
    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    left_elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
    left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
    left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
    right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    right_elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
    right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
    right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
    right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    left_knee_angle = calculate_angle(left_ankle, left_knee, left_hip)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
    right_knee_angle = calculate_angle(right_ankle, right_knee, right_hip)

    return {'left_elbow': left_elbow_angle, 'left_hip': left_hip_angle, 'left_shoulder': left_shoulder_angle,
            'left_knee': left_knee_angle,
            'right_elbow': right_elbow_angle, 'right_hip': right_hip_angle, 'right_shoulder': right_shoulder_angle,
            'right_knee': right_knee_angle}


def get_visible_side(landmarks):
    right_side_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility + landmarks[
        mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility + landmarks[
                                mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility + landmarks[
                                mp_pose.PoseLandmark.RIGHT_HIP.value].visibility + landmarks[
                                mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    left_side_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility + landmarks[
        mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility + landmarks[
                               mp_pose.PoseLandmark.LEFT_WRIST.value].visibility + landmarks[
                               mp_pose.PoseLandmark.LEFT_HIP.value].visibility + landmarks[
                               mp_pose.PoseLandmark.LEFT_KNEE.value].visibility

    return 'r' if right_side_visibility > left_side_visibility else 'l'
