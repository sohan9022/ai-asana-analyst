import numpy as np
from app.db.connection import get_connection

def load_rules(pose_name):
    """Load pose rules from the database for a given pose."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT joint_name, min_angle, max_angle, alignment_type
        FROM pose_rules
        WHERE pose_name = %s
    """, (pose_name,))
    rules = cursor.fetchall()
    cursor.close()
    conn.close()
    return rules

def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A, B, C.
    Each point is a (x, y) tuple from MediaPipe landmarks.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Avoid floating point errors outside of the [-1, 1] range for arccos
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)  
    angle = np.degrees(np.arccos(cosine))
    return round(angle, 2)

def check_alignment(landmarks, alignment_type):
    """
    Check if body parts are aligned horizontally or vertically.
    Returns (is_correct, message)
    """
    if alignment_type == "horizontal":
        # Check if shoulders are level (y coordinates should be close)
        left_shoulder  = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")

        if left_shoulder and right_shoulder:
            diff = abs(left_shoulder[1] - right_shoulder[1])
            if diff > 0.05:  # 5% of frame height tolerance
                return False, "Shoulders are not level — relax your shoulders down and keep them even"
            return True, "Shoulders are level"

    if alignment_type == "vertical":
        # Check if hips are level
        left_hip  = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")

        if left_hip and right_hip:
            diff = abs(left_hip[1] - right_hip[1])
            if diff > 0.05:
                return False, "Hips are not level — square your hips to the front"
            return True, "Hips are level"

    return True, "Alignment OK"

def get_correction(joint_name, detected_angle, min_angle, max_angle):
    """
    Return anatomically correct mistake and correction strings.
    In MediaPipe: 180° is a straight line. <90° is a sharp bend.
    'too_low' = angle is smaller than minimum (bent too much).
    'too_high' = angle is larger than maximum (too straight).
    """
    corrections = {
        "front_knee": {
            "too_low":  ("Front knee bent too deeply",
                         "Shift your weight back so your knee stacks directly over your ankle"),
            "too_high": ("Front knee not bent enough",
                         "Bend your front knee deeper until your thigh is parallel to the floor"),
        },
        "back_knee": {
            "too_low":  ("Back knee is bent",
                         "Straighten your back leg fully and press firmly through your outer heel"),
            "too_high": ("Back knee is hyperextended",
                         "Micro-bend your back knee slightly to avoid locking the joint"),
        },
        "standing_knee": {
            "too_low":  ("Standing leg is bent",
                         "Straighten your standing leg and engage your thigh muscles for stability"),
            "too_high": ("Standing knee is hyperextended",
                         "Keep a micro-bend in your standing knee to protect the joint"),
        },
        "raised_knee": {
            "too_low":  ("Raised knee angle too tight",
                         "Relax your raised leg slightly to ease the tension"),
            "too_high": ("Raised knee is dropping",
                         "Lift your raised foot higher up your inner thigh or calf"),
        },
        "hip": {
            "too_low":  ("Torso leaning forward",
                         "Lift your chest and stack your shoulders directly over your hips"),
            "too_high": ("Torso leaning backward",
                         "Bring your torso forward to center your weight over your hips"),
        },
    }

    # Determine which threshold was violated
    if detected_angle < min_angle:
        direction = "too_low"
    else:
        direction = "too_high"

    # Fallback string if joint isn't explicitly mapped above
    default = (
        f"{joint_name.replace('_', ' ').title()} angle out of range",
        f"Adjust your {joint_name.replace('_', ' ')} to be between {min_angle}° and {max_angle}°"
    )

    mistake, correction = corrections.get(joint_name, {}).get(direction, default)
    return mistake, correction

def run_constraint_check(pose_name, joint_angles, landmarks=None):
    """
    Main function — runs all constraint checks for a given pose.
    """
    rules = load_rules(pose_name)
    violations = []

    for joint_name, min_angle, max_angle, alignment_type in rules:

        # ── Angle check ───────────────────────────────────────────────
        if min_angle is not None and max_angle is not None:
            detected = joint_angles.get(joint_name)

            if detected is None:
                continue  # Landmark not detected, skip

            if not (min_angle <= detected <= max_angle):
                mistake, correction = get_correction(
                    joint_name, detected, min_angle, max_angle
                )
                violations.append({
                    "joint":      joint_name,
                    "detected":   detected,
                    "min":        min_angle,
                    "max":        max_angle,
                    "mistake":    mistake,
                    "correction": correction,
                    "type":       "angle"
                })

        # ── Alignment check ───────────────────────────────────────────
        if alignment_type and landmarks:
            is_correct, message = check_alignment(landmarks, alignment_type)
            if not is_correct:
                violations.append({
                    "joint":      joint_name,
                    "detected":   None,
                    "min":        None,
                    "max":        None,
                    "mistake":    message,
                    "correction": "Focus on keeping your body symmetrical and level",
                    "type":       "alignment"
                })

    return violations