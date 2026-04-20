# import cv2
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import av
# import numpy as np
# import tensorflow as tf
# import os
# import base64
# from google import genai
# from dotenv import load_dotenv
# from gtts import gTTS

# from app.engine.constraint_engine import run_constraint_check, calculate_angle
# import mediapipe as mp
# from mediapipe.tasks.python import vision as mp_vision
# from mediapipe.tasks.python.core import base_options as mp_base_options
# from mediapipe.tasks.python.vision.core import vision_task_running_mode
# from app.live.audio_feedback import AudioFeedback

# # ADDED: Import your PDF generator
# from app.report.report_generator import generate_report 

# # ── SETUP GEMINI API ──────────────────────────────────────────────────────────
# load_dotenv()
# client = genai.Client()

# # ── AI VOICE COACH FUNCTIONS ──────────────────────────────────────────────────
# def get_live_voice_coaching(pose_name, violations):
#     """Asks Gemini for a strict, continuous paragraph based on current mistakes."""
#     if not violations:
#         return f"Your {pose_name} looks excellent! Hold that alignment, keep your core engaged, and remember to take deep breaths."
    
#     # Extract just the text mistakes to keep the prompt clean
#     mistakes_list = [v['mistake'] for v in violations]
#     mistakes_str = ", ".join(mistakes_list)
    
#     prompt = f"""
#     Act as a professional Yoga Audio Coach. The user is holding {pose_name} but making these mistakes: {mistakes_str}.
    
#     Write EXACTLY THREE SENTENCES to speak out loud to them right now to fix their pose. 
#     Sentence 1: Acknowledge the pose and the main issue.
#     Sentence 2: Give a direct physical instruction to fix it.
#     Sentence 3: An encouraging cue about breathing or balance.
    
#     CRITICAL INSTRUCTION: Return ONLY the spoken words. Do not use line breaks, bullet points, asterisks, bold text, or emojis. Write it as a single, continuous paragraph.
#     """
#     try:
#         model = genai.GenerativeModel('gemini-1.5-flash')
#         response = client.models.generate_content(
#             model='gemini-3.1-flash-lite', # Using the fast, high-limit model we discussed!
#             contents=prompt
#         )
#         # Strip out any rogue markdown just in case Gemini disobeys
#         clean_text = response.text.replace("*", "").replace("#", "").replace("\n", " ")
#         return clean_text
#     except Exception as e:
#         return "Keep holding the pose, check your alignment on the screen, and breathe deeply."


# def play_audio_from_text(text):
#     """Converts text to speech and plays it using Streamlit's stable native player."""
#     tts = gTTS(text=text, lang='en', tld='co.uk')
#     audio_file = "coach_audio.mp3"
#     tts.save(audio_file)
    
#     with open(audio_file, "rb") as f:
#         audio_bytes = f.read()
        
#     # Streamlit's native audio widget handles browser autoplay much better!
#     st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    
#     os.remove(audio_file)
# # ── LOAD KERAS CNN MODEL ──────────────────────────────────────────────────────
# @st.cache_resource 
# def load_pose_classifier():
#     try:
#         model = tf.keras.models.load_model("app/engine/yoga_pose_model.keras")
#         return model
#     except Exception as e:
#         st.error(f"Failed to load CNN model: {e}")
#         return None

# POSE_MODEL = load_pose_classifier()

# KERAS_CLASS_MAP = {
#     0: "Downward Dog", 
#     1: "Goddess Pose", 
#     2: "Plank", 
#     3: "Tree Pose", 
#     4: "Warrior II"
# }

# # ── Landmark indices ──────────────────────────────────────────────────────────
# LM = {
#     "nose":            0,
#     "left_shoulder":   11,
#     "right_shoulder":  12,
#     "left_elbow":      13,
#     "right_elbow":     14,
#     "left_wrist":      15,
#     "right_wrist":     16,
#     "left_hip":        23,
#     "right_hip":       24,
#     "left_knee":       25,
#     "right_knee":      26,
#     "left_ankle":      27,
#     "right_ankle":     28,
# }

# def get_coords(landmarks, name, w, h):
#     lm = landmarks[LM[name]]
#     return (int(lm.x * w), int(lm.y * h))

# def get_norm(landmarks, name):
#     lm = landmarks[LM[name]]
#     return (lm.x, lm.y)

# def extract_joint_angles(landmarks, w, h):
#     def c(name): return get_coords(landmarks, name, w, h)

#     if c("left_ankle")[0] < c("right_ankle")[0]:
#         front_h, front_k, front_a = "left_hip", "left_knee", "left_ankle"
#         back_h, back_k, back_a = "right_hip", "right_knee", "right_ankle"
#     else:
#         front_h, front_k, front_a = "right_hip", "right_knee", "right_ankle"
#         back_h, back_k, back_a = "left_hip", "left_knee", "left_ankle"

#     if c("left_knee")[1] < c("right_knee")[1]:
#         raised_h, raised_k, raised_a = "left_hip", "left_knee", "left_ankle"
#         stand_h, stand_k, stand_a = "right_hip", "right_knee", "right_ankle"
#     else:
#         raised_h, raised_k, raised_a = "right_hip", "right_knee", "right_ankle"
#         stand_h, stand_k, stand_a = "left_hip", "left_knee", "left_ankle"

#     angles = {}
#     angles["front_knee"]    = calculate_angle(c(front_h), c(front_k), c(front_a))
#     angles["back_knee"]     = calculate_angle(c(back_h), c(back_k), c(back_a))
#     angles["standing_knee"] = calculate_angle(c(stand_h), c(stand_k), c(stand_a))
#     angles["raised_knee"]   = calculate_angle(c(raised_h), c(raised_k), c(raised_a))
#     angles["hip"]           = calculate_angle(c("left_shoulder"), c("left_hip"), c("left_knee"))
    
#     return angles

# def extract_landmarks_for_alignment(landmarks):
#     return {
#         "left_shoulder":  get_norm(landmarks, "left_shoulder"),
#         "right_shoulder": get_norm(landmarks, "right_shoulder"),
#         "left_hip":       get_norm(landmarks, "left_hip"),
#         "right_hip":      get_norm(landmarks, "right_hip"),
#     }

# def draw_skeleton(frame, landmarks, violations, w, h):
#     violated_joints = {v["joint"] for v in violations if v["type"] == "angle"}

#     connections = [
#         ("left_shoulder",  "left_elbow"), ("left_elbow",     "left_wrist"),
#         ("right_shoulder", "right_elbow"), ("right_elbow",    "right_wrist"),
#         ("left_shoulder",  "right_shoulder"), ("left_shoulder",  "left_hip"),
#         ("right_shoulder", "right_hip"), ("left_hip",       "right_hip"),
#         ("left_hip",       "left_knee"), ("left_knee",      "left_ankle"),
#         ("right_hip",      "right_knee"), ("right_knee",     "right_ankle"),
#     ]

#     for start_name, end_name in connections:
#         start = get_coords(landmarks, start_name, w, h)
#         end   = get_coords(landmarks, end_name,   w, h)
#         is_violated = any(j in start_name or j in end_name for j in violated_joints)
#         color = (0, 0, 255) if is_violated else (0, 255, 0)
#         cv2.line(frame, start, end, color, 3)

#     for name in LM:
#         cv2.circle(frame, get_coords(landmarks, name, w, h), 5, (255, 255, 255), -1)

#     return frame

# def draw_feedback_panel(frame, violations, pose_name):
#     h, w = frame.shape[:2]
#     overlay = frame.copy()
    
#     box_width = 380
#     box_height = 80 + (len(violations[:3]) * 60) if violations else 80
#     start_x = w - box_width - 20 
#     start_y = 20                 
    
#     cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

#     y = start_y + 35
#     cv2.putText(frame, f"Pose: {pose_name}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#     y += 35

#     if not violations:
#         cv2.putText(frame, "Perfect form! Hold it.", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     else:
#         for v in violations[:3]:
#             cv2.putText(frame, f"X {v['mistake'][:40]}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#             y += 20
#             words = v["correction"].split()
#             line  = ""
#             for word in words:
#                 if len(line + word) < 40:
#                     line += word + " "
#                 else:
#                     cv2.putText(frame, f"  {line}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
#                     y += 20
#                     line = word + " "
#             if line:
#                 cv2.putText(frame, f"  {line}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
#                 y += 25
                
#     return frame

# # ── Video Processor ───────────────────────────────────────────────────────────
# class PoseProcessor(VideoProcessorBase):
#     def __init__(self):
#         BaseOptions = mp_base_options.BaseOptions
#         RunningMode = vision_task_running_mode.VisionTaskRunningMode

#         options = mp_vision.PoseLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
#             running_mode=RunningMode.VIDEO,
#             num_poses=1,
#             min_pose_detection_confidence=0.5,
#             min_tracking_confidence=0.5,
#         )
#         self.detector   = mp_vision.PoseLandmarker.create_from_options(options)
        
#         self.mode = "Auto-Detect (AI)" 
#         self.pose_name  = "Warrior II"
#         self.violations = []
        
#         self.frame_ts   = 0
#         self.frame_count = 0
#         self.inference_interval = 30 
#         self.audio      = AudioFeedback()  
        
#         # ADDED: A list to save the session history
#         self.history = []

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         h, w = img.shape[:2]

#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
#         self.frame_ts += 1
#         results = self.detector.detect_for_video(mp_image, self.frame_ts)
#         self.frame_count += 1

#         if results.pose_landmarks and len(results.pose_landmarks) > 0:
#             landmarks = results.pose_landmarks[0]

#             if self.mode == "Auto-Detect (AI)":
#                 if POSE_MODEL is not None and self.frame_count % self.inference_interval == 0:
#                     x_coords = [int(lm.x * w) for lm in landmarks]
#                     y_coords = [int(lm.y * h) for lm in landmarks]
#                     x_min, x_max = max(0, min(x_coords) - 50), min(w, max(x_coords) + 50)
#                     y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 50)
                    
#                     cropped_rgb = rgb[y_min:y_max, x_min:x_max]
#                     input_img = cv2.resize(cropped_rgb, (75, 75))
#                     input_img = input_img.astype(np.float32) / 255.0
#                     input_img = np.expand_dims(input_img, axis=0)
                    
#                     predictions = POSE_MODEL.predict(input_img, verbose=0)
#                     class_idx = np.argmax(predictions[0])
#                     confidence = np.max(predictions[0]) * 100
                    
#                     if confidence > 60.0:
#                         self.pose_name = KERAS_CLASS_MAP.get(class_idx, self.pose_name)
#             else:
#                 self.pose_name = self.mode

#             joint_angles = extract_joint_angles(landmarks, w, h)
#             alignment_lm = extract_landmarks_for_alignment(landmarks)

#             self.violations = run_constraint_check(self.pose_name, joint_angles, alignment_lm)

#             # Local audio beep feedback
#             self.audio.process_violations(self.violations)

#             # Visual feedback
#             img = draw_skeleton(img, landmarks, self.violations, w, h)
#             img = draw_feedback_panel(img, self.violations, self.pose_name)

#             # ADDED: Save a snapshot every ~1 second for the PDF report
#             if self.frame_count % self.inference_interval == 0:
#                 self.history.append((img.copy(), self.violations, joint_angles, "ok"))

#         else:
#             self.violations = [] # Clear violations if nobody is in frame
#             cv2.putText(img, "No pose detected — adjust position", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# # ── Streamlit UI ──────────────────────────────────────────────────────────────
# def run_live_mode():
#     st.title("AI Asana Analyst — Live Coach")
#     st.markdown("Select your pose or use AI Auto-Detect, step back, and the AI will guide your alignment in real-time.")

#     # Top control row
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         selected_mode = st.selectbox(
#             "Select Practice Mode",
#             ["Auto-Detect (AI)", "Warrior II", "Tree Pose", "Downward Dog", "Plank", "Goddess Pose", "Triangle Pose"]
#         )
#     with col2:
#         if "coach_muted" not in st.session_state:
#             st.session_state.coach_muted = False
#         st.write("") 
#         st.write("") 
#         mute_toggle = st.checkbox("🔇 Mute Local Beeps", value=st.session_state.coach_muted)
#         st.session_state.coach_muted = mute_toggle

#     guidance = {
#         "Auto-Detect (AI)": "📷 AI will detect your pose. Face the camera directly or stand side-on.",
#         "Warrior II":    "📷 Camera angle: Side profile view",
#         "Tree Pose":     "📷 Camera angle: Front facing view",
#         "Downward Dog":  "📷 Camera angle: Side profile view",
#         "Plank":         "📷 Camera angle: Side profile view",
#         "Goddess Pose":  "📷 Camera angle: Front facing view",
#         "Triangle Pose": "📷 Camera angle: Side profile view",
#     }
#     st.info(guidance.get(selected_mode, ""))

#     # Video and AI Coach Layout
#     video_col, ai_col = st.columns([2, 1])
    
#     with video_col:
#         ctx = webrtc_streamer(
#             key="pose",
#             video_processor_factory=PoseProcessor,
#             rtc_configuration={
#                 "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#             },
#             media_stream_constraints={
#                 "video": {
#                     "width": {"ideal": 640},  
#                     "height": {"ideal": 480},
#                     "frameRate": {"ideal": 15, "max": 20} 
#                 }, 
#                 "audio": False
#             },
#             async_processing=True, 
#         )

#         if ctx.video_processor:
#             ctx.video_processor.mode = selected_mode
#             ctx.video_processor.audio.muted = st.session_state.coach_muted

#     with ai_col:
#         st.markdown("### 🤖 Ask AI Coach")
#         st.write("Need help? Click below and the AI will analyze your current form and speak to you out loud.")
        
#         if ctx.video_processor:
#             if st.button("🗣️ Coach Me Now", type="primary", use_container_width=True):
#                 with st.spinner("AI is analyzing your pose..."):
#                     current_pose = ctx.video_processor.pose_name
#                     current_violations = ctx.video_processor.violations
                    
#                     # 1. Get the script from Gemini
#                     ai_script = get_live_voice_coaching(current_pose, current_violations)
                    
#                     # 2. Display the text
#                     st.success(f"**Coach says:** {ai_script}")
                    
#                     # 3. Read it out loud!
#                     play_audio_from_text(ai_script)
#         else:
#             st.warning("Start the video feed to enable the AI Coach.")

#     # ADDED SECTION: The PDF Generator block at the bottom
#     st.divider()
#     st.markdown("### 📊 Session Complete?")
#     st.write("When you are done with your practice, click below to generate your personalized AI Posture Report.")
    
#     if ctx.video_processor:
#         if st.button("End Session & Generate PDF Report", type="primary"):
            
#             # Check if they actually held a pose long enough to get data
#             if len(ctx.video_processor.history) > 0:
#                 with st.spinner("Gemini AI is writing your Yoga Wisdom summary and building the PDF..."):
                    
#                     report_path = "live_session_report.pdf"
#                     final_pose = ctx.video_processor.pose_name
#                     session_data = ctx.video_processor.history
                    
#                     # Call your existing PDF generator!
#                     generate_report(final_pose, session_data, report_path)
                    
#                     st.success("Your AI Report is ready!")
                    
#                     # Provide a Streamlit download button for the file
#                     with open(report_path, "rb") as pdf_file:
#                         st.download_button(
#                             label="📥 Download PDF Report",
#                             data=pdf_file,
#                             file_name="My_Yoga_Session_Report.pdf",
#                             mime="application/pdf"
#                         )
#             else:
#                 st.warning("No session data collected yet. Please stand in front of the camera and hold a pose first!")



import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
import tensorflow as tf
import os
import base64
import urllib.request

# Local Engine & AI Imports
from app.engine.constraint_engine import run_constraint_check, calculate_angle
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from app.live.audio_feedback import AudioFeedback
from app.report.report_generator import generate_report 


# ── LOAD KERAS CNN MODEL ──────────────────────────────────────────────────────
@st.cache_resource 
def load_pose_classifier():
    model_path = "app/engine/yoga_pose_model.keras"
    
    # 👇 CHANGE THESE TWO LINES to match your exact GitHub details
    GITHUB_USER = os.getenv("GITHUB_USER")
    GITHUB_REPO = os.getenv("GITHUB_REPO")
    
    # This magic URL automatically grabs the file from your newest Release
    github_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/latest/download/yoga_pose_model.keras"

    # 1. If the model isn't on your computer, download it!
    if not os.path.exists(model_path):
        st.info("Downloading latest AI model from GitHub... please wait.")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(github_url, model_path)
            st.success("✅ Download complete! Starting AI Coach...")
        except Exception as e:
            st.error(f"Failed to download model. Ensure your username and repo are correct. Error: {e}")
            return None

    # 2. Load the model into the app
    try:
        model = tf.keras.models.load_model(model_path)
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

    if c("left_ankle")[0] < c("right_ankle")[0]:
        front_h, front_k, front_a = "left_hip", "left_knee", "left_ankle"
        back_h, back_k, back_a = "right_hip", "right_knee", "right_ankle"
    else:
        front_h, front_k, front_a = "right_hip", "right_knee", "right_ankle"
        back_h, back_k, back_a = "left_hip", "left_knee", "left_ankle"

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
    
    box_width = 380
    box_height = 80 + (len(violations[:3]) * 60) if violations else 80
    start_x = w - box_width - 20 
    start_y = 20                
    
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = start_y + 35
    cv2.putText(frame, f"Pose: {pose_name}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 35

    if not violations:
        cv2.putText(frame, "Perfect form! Hold it.", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        for v in violations[:3]:
            cv2.putText(frame, f"X {v['mistake'][:40]}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 20
            words = v["correction"].split()
            line  = ""
            for word in words:
                if len(line + word) < 40:
                    line += word + " "
                else:
                    cv2.putText(frame, f"  {line}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    y += 20
                    line = word + " "
            if line:
                cv2.putText(frame, f"  {line}", (start_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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
        
        self.mode = "Auto-Detect (AI)" 
        self.pose_name  = "Warrior II"
        self.violations = []
        
        self.frame_ts   = 0
        self.frame_count = 0
        self.inference_interval = 30 
        
        # Audio daemon initialized here
        self.audio      = AudioFeedback()  
        
        # Save session history for PDF report
        self.history = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_ts += 1
        results = self.detector.detect_for_video(mp_image, self.frame_ts)
        self.frame_count += 1

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]

            if self.mode == "Auto-Detect (AI)":
                if POSE_MODEL is not None and self.frame_count % self.inference_interval == 0:
                    x_coords = [int(lm.x * w) for lm in landmarks]
                    y_coords = [int(lm.y * h) for lm in landmarks]
                    x_min, x_max = max(0, min(x_coords) - 50), min(w, max(x_coords) + 50)
                    y_min, y_max = max(0, min(y_coords) - 50), min(h, max(y_coords) + 50)
                    
                    cropped_rgb = rgb[y_min:y_max, x_min:x_max]
                    input_img = cv2.resize(cropped_rgb, (75, 75))
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)
                    
                    predictions = POSE_MODEL.predict(input_img, verbose=0)
                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    
                    if confidence > 60.0:
                        self.pose_name = KERAS_CLASS_MAP.get(class_idx, self.pose_name)
            else:
                self.pose_name = self.mode

            joint_angles = extract_joint_angles(landmarks, w, h)
            alignment_lm = extract_landmarks_for_alignment(landmarks)

            self.violations = run_constraint_check(self.pose_name, joint_angles, alignment_lm)

            # Local background audio feedback (Non-blocking)
            self.audio.process_violations(self.violations)

            # Visual feedback
            img = draw_skeleton(img, landmarks, self.violations, w, h)
            img = draw_feedback_panel(img, self.violations, self.pose_name)

            # Save snapshot every ~1 second for PDF report
            if self.frame_count % self.inference_interval == 0:
                self.history.append((img.copy(), self.violations, joint_angles, "ok"))

        else:
            self.violations = [] # Clear violations if nobody is in frame
            cv2.putText(img, "No pose detected — adjust position", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
def run_live_mode():
    st.title("AI Asana Analyst — Live Coach")
    st.markdown("Select your pose or use AI Auto-Detect, step back, and the AI will guide your alignment in real-time.")

    # Top control row
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_mode = st.selectbox(
            "Select Practice Mode",
            ["Auto-Detect (AI)", "Warrior II", "Tree Pose", "Downward Dog", "Plank", "Goddess Pose", "Triangle Pose"]
        )
    with col2:
        if "coach_muted" not in st.session_state:
            st.session_state.coach_muted = False
        st.write("") 
        st.write("") 
        mute_toggle = st.checkbox("🔇 Mute Audio Coach", value=st.session_state.coach_muted)
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

    # Video and AI Coach Layout
    video_col, ai_col = st.columns([2, 1])
    
    with video_col:
        ctx = webrtc_streamer(
            key="pose",
            video_processor_factory=PoseProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},  
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 20} 
                }, 
                "audio": False
            },
            async_processing=True, 
        )

        if ctx.video_processor:
            ctx.video_processor.mode = selected_mode
            ctx.video_processor.audio.muted = st.session_state.coach_muted

    with ai_col:
        st.markdown("### 🤖 Live Audio Coach")
        st.write("The AI is continuously analyzing your joints.")
        
        if ctx.state.playing:
            st.success("🟢 Coach is active! Move into the frame.")
            st.write("If you break a biomechanical rule, the coach will instantly speak out loud to correct you.")
            st.info("Watch the video overlay to see real-time geometry tracking.")
        else:
            st.warning("Start the video feed to activate the automatic AI Coach.")


    # The PDF Generator block at the bottom
    st.divider()
    st.markdown("### 📊 Session Complete?")
    st.write("When you are done with your practice, click below to generate your personalized AI Posture Report.")
    
    if ctx.video_processor:
        if st.button("End Session & Generate PDF Report", type="primary"):
            
            # Check if they actually held a pose long enough to get data
            if len(ctx.video_processor.history) > 0:
                with st.spinner("Gemini AI is writing your Yoga Wisdom summary and building the PDF..."):
                    
                    report_path = "live_session_report.pdf"
                    final_pose = ctx.video_processor.pose_name
                    session_data = ctx.video_processor.history
                    
                    # Generate the async cloud report
                    generate_report(final_pose, session_data, report_path)
                    
                    st.success("Your AI Report is ready!")
                    
                    # Provide a Streamlit download button for the file
                    with open(report_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_file,
                            file_name="My_Yoga_Session_Report.pdf",
                            mime="application/pdf"
                        )
            else:
                st.warning("No session data collected yet. Please stand in front of the camera and hold a pose first!")