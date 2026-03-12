import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from app.engine.constraint_engine import run_constraint_check, calculate_angle

# ── Landmark indices ──────────────────────────────────────────────────────────
LM = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24, "left_knee": 25,
    "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}

def get_coords(landmarks, name, w, h):
    lm = landmarks[LM[name]]
    return (int(lm.x * w), int(lm.y * h))

def get_norm(landmarks, name):
    lm = landmarks[LM[name]]
    return (lm.x, lm.y)

def build_detector():
    """Build and return a MediaPipe pose landmarker for images."""
    BaseOptions = mp_base_options.BaseOptions
    options = mp_vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def extract_joint_angles(landmarks, w, h):
    def c(name): return get_coords(landmarks, name, w, h)
    
    # Dynamically determine which leg is in front based on X coordinate
    # (Assuming side-profile view for Warrior II / Triangle)
    left_ankle_x = c("left_ankle")[0]
    right_ankle_x = c("right_ankle")[0]
    
    if left_ankle_x < right_ankle_x: # Left leg is forward (closer to x=0)
        front_hip, front_knee, front_ankle = "left_hip", "left_knee", "left_ankle"
        back_hip, back_knee, back_ankle = "right_hip", "right_knee", "right_ankle"
    else: # Right leg is forward
        front_hip, front_knee, front_ankle = "right_hip", "right_knee", "right_ankle"
        back_hip, back_knee, back_ankle = "left_hip", "left_knee", "left_ankle"

    return {
        "front_knee":    calculate_angle(c(front_hip), c(front_knee), c(front_ankle)),
        "back_knee":     calculate_angle(c(back_hip), c(back_knee), c(back_ankle)),
        # For Tree Pose (front-facing), we can assume raised leg is the one with the knee higher up (smaller Y)
        "standing_knee": calculate_angle(c("right_hip"), c("right_knee"), c("right_ankle")) if c("right_knee")[1] > c("left_knee")[1] else calculate_angle(c("left_hip"), c("left_knee"), c("left_ankle")),
        "raised_knee":   calculate_angle(c("left_hip"), c("left_knee"), c("left_ankle")) if c("left_knee")[1] < c("right_knee")[1] else calculate_angle(c("right_hip"), c("right_knee"), c("right_ankle")),
        "hip":           calculate_angle(c("left_shoulder"), c("left_hip"), c("left_knee")), # You may want to make this dynamic too!
    }

def extract_landmarks_for_alignment(landmarks):
    return {
        "left_shoulder":  get_norm(landmarks, "left_shoulder"),
        "right_shoulder": get_norm(landmarks, "right_shoulder"),
        "left_hip":       get_norm(landmarks, "left_hip"),
        "right_hip":      get_norm(landmarks, "right_hip"),
    }

def annotate_frame(frame, landmarks, violations, w, h):
    """Draw skeleton and highlight violated joints on the frame."""
    violated_joints = {v["joint"] for v in violations if v["type"] == "angle"}

    connections = [
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    for s, e in connections:
        start = get_coords(landmarks, s, w, h)
        end   = get_coords(landmarks, e, w, h)
        is_violated = any(j in s or j in e for j in violated_joints)
        color = (0, 0, 255) if is_violated else (0, 255, 0)
        cv2.line(frame, start, end, color, 3)

    for name in LM:
        cv2.circle(frame, get_coords(landmarks, name, w, h), 5, (255, 255, 255), -1)

    return frame

def analyze_image(image_path, pose_name):
    """
    Analyze a single image.
    Returns: (annotated_frame, violations, joint_angles)
    """
    frame    = cv2.imread(image_path)
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detector = build_detector()
    results  = detector.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return frame, [], {}, "undetectable"

    landmarks    = results.pose_landmarks[0]
    joint_angles = extract_joint_angles(landmarks, w, h)
    alignment_lm = extract_landmarks_for_alignment(landmarks)
    violations   = run_constraint_check(pose_name, joint_angles, alignment_lm)
    annotated    = annotate_frame(frame.copy(), landmarks, violations, w, h)

    return annotated, violations, joint_angles, "ok"

def analyze_video(video_path, pose_name, sample_every=10):
    """
    Analyze a video file by sampling every N frames.
    Returns: list of (frame, violations, joint_angles, status)
    """
    cap     = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0

    detector = build_detector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_every != 0:
            continue

        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            results.append((frame, [], {}, "undetectable"))
            continue

        landmarks    = result.pose_landmarks[0]
        joint_angles = extract_joint_angles(landmarks, w, h)
        alignment_lm = extract_landmarks_for_alignment(landmarks)
        violations   = run_constraint_check(pose_name, joint_angles, alignment_lm)
        annotated    = annotate_frame(frame.copy(), landmarks, violations, w, h)
        results.append((annotated, violations, joint_angles, "ok"))

    cap.release()
    return results