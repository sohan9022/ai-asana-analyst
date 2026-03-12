import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
import tensorflow as tf

from app.engine.constraint_engine import run_constraint_check, calculate_angle
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from app.live.audio_feedback import AudioFeedback

# ── LOAD KERAS CNN MODEL ──────────────────────────────────────────────────────
@st.cache_resource 
def load_pose_classifier():
    try:
        model = tf.keras.models.load_model("app/engine/yoga_pose_model.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load CNN model: {e}")
        return None

POSE_MODEL = load_pose_classifier()

KERAS_CLASS_MAP = {
    0: "Downward Dog", 
    1: "Goddess Pose", 
    2: "Plank", 
    3: "Tree Pose", 
    4: "Warrior II"
}

# ── Landmark indices ──────────────────────────────────────────────────────────
LM = {
    "nose":            0,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
}

def get_coords(landmarks, name, w, h):
    lm = landmarks[LM[name]]
    return (int(lm.x * w), int(lm.y * h))

def get_norm(landmarks, name):
    lm = landmarks[LM[name]]
    return (lm.x, lm.y)

def extract_joint_angles(landmarks, w, h):
    def c(name): return get_coords(landmarks, name, w, h)

    # ── DYNAMIC LEG DETECTION ──
    # Determine which leg is "front" based on X-coordinates (closer to left edge)
    if c("left_ankle")[0] < c("right_ankle")[0]:
        front_h, front_k, front_a = "left_hip", "left_knee", "left_ankle"
        back_h, back_k, back_a = "right_hip", "right_knee", "right_ankle"
    else:
        front_h, front_k, front_a = "right_hip", "right_knee", "right_ankle"
        back_h, back_k, back_a = "left_hip", "left_knee", "left_ankle"

    # Determine which leg is "raised" for Tree Pose based on Y-coordinates (higher up)
    if c("left_knee")[1] < c("right_knee")[1]:
        raised_h, raised_k, raised_a = "left_hip", "left_knee", "left_ankle"
        stand_h, stand_k, stand_a = "right_hip", "right_knee", "right_ankle"
    else:
        raised_h, raised_k, raised_a = "right_hip", "right_knee", "right_ankle"
        stand_h, stand_k, stand_a = "left_hip", "left_knee", "left_ankle"

    angles = {}
    angles["front_knee"]    = calculate_angle(c(front_h), c(front_k), c(front_a))
    angles["back_knee"]     = calculate_angle(c(back_h), c(back_k), c(back_a))
    angles["standing_knee"] = calculate_angle(c(stand_h), c(stand_k), c(stand_a))
    angles["raised_knee"]   = calculate_angle(c(raised_h), c(raised_k), c(raised_a))
    angles["hip"]           = calculate_angle(c("left_shoulder"), c("left_hip"), c("left_knee"))
    
    return angles

def extract_landmarks_for_alignment(landmarks):
    return {
        "left_shoulder":  get_norm(landmarks, "left_shoulder"),
        "right_shoulder": get_norm(landmarks, "right_shoulder"),
        "left_hip":       get_norm(landmarks, "left_hip"),
        "right_hip":      get_norm(landmarks, "right_hip"),
    }

def draw_skeleton(frame, landmarks, violations, w, h):
    violated_joints = {v["joint"] for v in violations if v["type"] == "angle"}

    connections = [
        ("left_shoulder",  "left_elbow"), ("left_elbow",     "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow",    "right_wrist"),
        ("left_shoulder",  "right_shoulder"), ("left_shoulder",  "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip",       "right_hip"),
        ("left_hip",       "left_knee"), ("left_knee",      "left_ankle"),
        ("right_hip",      "right_knee"), ("right_knee",     "right_ankle"),
    ]

    for start_name, end_name in connections:
        start = get_coords(landmarks, start_name, w, h)
        end   = get_coords(landmarks, end_name,   w, h)
        is_violated = any(j in start_name or j in end_name for j in violated_joints)
        color = (0, 0, 255) if is_violated else (0, 255, 0)
        cv2.line(frame, start, end, color, 3)

    for name in LM:
        cv2.circle(frame, get_coords(landmarks, name, w, h), 5, (255, 255, 255), -1)

    return frame

def draw_feedback_panel(frame, violations, pose_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # ── UI SETTINGS: Small floating window at the top right ──
    box_width = 380
    # Dynamically calculate height based on the number of mistakes (or keep it small if perfect)
    box_height = 80 + (len(violations[:3]) * 60) if violations else 80
    
    start_x = w - box_width - 20 # 20 pixels of padding from the right edge
    start_y = 20                 # 20 pixels of padding from the top edge
    
    # Draw the semi-transparent black background
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # ── DRAW THE POSE NAME HEADER ──
    y = start_y + 35
    cv2.putText(frame, f"Pose: {pose_name}", (start_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 35

    # ── DRAW THE TEXT (MISTAKES OR PRAISE) ──
    if not violations:
        cv2.putText(frame, "Perfect form! Hold it.", (start_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Only show the top 3 mistakes so the box doesn't run off the screen
        for v in violations[:3]:
            # Print the Mistake in Red
            cv2.putText(frame, f"X {v['mistake'][:40]}", (start_x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 20
            
            # Print the Correction in Cyan (Yellow-ish) and wrap the text
            words = v["correction"].split()
            line  = ""
            for word in words:
                if len(line + word) < 40:
                    line += word + " "
                else:
                    cv2.putText(frame, f"  {line}", (start_x + 15, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y += 20
                    line = word + " "
            if line:
                cv2.putText(frame, f"  {line}", (start_x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                y += 25
                
    return frame

# ── Video Processor ───────────────────────────────────────────────────────────
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        BaseOptions = mp_base_options.BaseOptions
        RunningMode = vision_task_running_mode.VisionTaskRunningMode

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector   = mp_vision.PoseLandmarker.create_from_options(options)
        
        # State variables
        self.mode = "Auto-Detect (AI)" 
        self.pose_name  = "Warrior II"
        self.violations = []
        
        # Performance & Audio
        self.frame_ts   = 0
        self.frame_count = 0
        self.inference_interval = 30 # Run CNN every 30 frames
        self.audio      = AudioFeedback()  

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ── 1. MEDIAPIPE RUNS FIRST ──
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_ts += 1
        results = self.detector.detect_for_video(mp_image, self.frame_ts)

        self.frame_count += 1

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]

            # ── 2. HYBRID MODE LOGIC (THE SMART CROP) ──
            if self.mode == "Auto-Detect (AI)":
                if POSE_MODEL is not None and self.frame_count % self.inference_interval == 0:
                    
                    # Find the bounding box of the user's body
                    x_coords = [int(lm.x * w) for lm in landmarks]
                    y_coords = [int(lm.y * h) for lm in landmarks]
                    
                    # Add a 50-pixel margin around the body so we don't cut off limbs
                    x_min, x_max = max(0, min(x_coords) - 50), min(w, max(x_coords) + 50)
                    y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 50)
                    
                    # Crop the background out!
                    cropped_rgb = rgb[y_min:y_max, x_min:x_max]
                    
                    # Now send the clean, cropped image to your CNN
                    # Note: Ensure (75, 75) matches exactly what you used in your Jupyter notebook!
                    input_img = cv2.resize(cropped_rgb, (75, 75))
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)
                    
                    predictions = POSE_MODEL.predict(input_img, verbose=0)
                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    
                    # Only change the pose if the AI is reasonably confident (> 60%)
                    if confidence > 60.0:
                        self.pose_name = KERAS_CLASS_MAP.get(class_idx, self.pose_name)
            else:
                # Strictly use the manually selected pose
                self.pose_name = self.mode

            # ── 3. CONSTRAINT ENGINE CHECKS ──
            joint_angles = extract_joint_angles(landmarks, w, h)
            alignment_lm = extract_landmarks_for_alignment(landmarks)

            self.violations = run_constraint_check(
                self.pose_name, joint_angles, alignment_lm
            )

            # Audio coaching
            self.audio.process_violations(self.violations)

            # Visual coaching
            img = draw_skeleton(img, landmarks, self.violations, w, h)
            img = draw_feedback_panel(img, self.violations, self.pose_name)

        else:
            cv2.putText(img, "No pose detected — adjust position",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
# ── Streamlit UI ──────────────────────────────────────────────────────────────
def run_live_mode():
    st.title("AI Asana Analyst — Live Coach")
    st.markdown("Select your pose or use AI Auto-Detect, step back, and the AI will guide your alignment in real-time.")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_mode = st.selectbox(
            "Select Practice Mode",
            [
                "Auto-Detect (AI)", 
                "Warrior II", 
                "Tree Pose", 
                "Downward Dog", 
                "Plank", 
                "Goddess Pose", 
                "Triangle Pose"
            ]
        )
    with col2:
        # Use session state so checking the box doesn't break the WebRTC video stream
        if "coach_muted" not in st.session_state:
            st.session_state.coach_muted = False
        st.write("") # spacing
        st.write("") # spacing
        mute_toggle = st.checkbox("🔇 Mute Coach", value=st.session_state.coach_muted)
        st.session_state.coach_muted = mute_toggle

    guidance = {
        "Auto-Detect (AI)": "📷 AI will detect your pose. Face the camera directly or stand side-on.",
        "Warrior II":    "📷 Camera angle: Side profile view",
        "Tree Pose":     "📷 Camera angle: Front facing view",
        "Downward Dog":  "📷 Camera angle: Side profile view",
        "Plank":         "📷 Camera angle: Side profile view",
        "Goddess Pose":  "📷 Camera angle: Front facing view",
        "Triangle Pose": "📷 Camera angle: Side profile view",
    }
    st.info(guidance.get(selected_mode, ""))

    # ── THE LAG FIX: Optimize Camera Resolution and Processing ──
    ctx = webrtc_streamer(
        key="pose",
        video_processor_factory=PoseProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},   # Force a lighter 480p resolution
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 20} # Cap the framerate so Python can keep up
            }, 
            "audio": False
        },
        async_processing=True, # Critical: Runs the video processing in a background thread
    )

    if ctx.video_processor:
        # Pass the selected mode down to the processor
        ctx.video_processor.mode = selected_mode
        # Pass the mute state down to the audio engine
        ctx.video_processor.audio.muted = st.session_state.coach_muted