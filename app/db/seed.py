from app.db.connection import get_connection

def seed_pose_rules():
    conn = get_connection()
    cursor = conn.cursor()

    # Angles based on standard Hatha Yoga alignment:
    # 180 = Straight, 90 = Right Angle, <90 = Deep Bend
    rules = [
        # 1. Warrior II (Side View)
        # Front knee should be as close to a right angle as possible (90°).
        ("Warrior II", "front_knee",  85, 105, None, "side"),
        # Back leg must be strong and straight, but not locked.
        ("Warrior II", "back_knee",   165, 180, None, "side"),
        ("Warrior II", "shoulders",   None, None, "horizontal", "side"),

        # 2. Tree Pose (Front View)
        # Standing leg is straight.
        ("Tree Pose", "standing_knee", 170, 180, None, "front"),
        # Raised knee should be pushed back; angle varies by foot height (45-130°).
        ("Tree Pose", "raised_knee",    40, 130, None, "front"),
        ("Tree Pose", "hips",          None, None, "vertical", "front"),

        # 3. Downward Dog (Side View)
        # Focus on the 'V' shape. Hip angle is sharp (70-100°).
        ("Downward Dog", "hip",         70, 105, None, "side"),
        # Knees should be straight to stretch hamstrings.
        ("Downward Dog", "front_knee", 165, 180, None, "side"),
        ("Downward Dog", "back_knee",  165, 180, None, "side"),

        # 4. Plank (Side View)
        # The body must be a single straight line from head to heels.
        ("Plank", "hip",               165, 180, None, "side"),
        # Shoulders must be level.
        ("Plank", "shoulders",         None, None, "horizontal", "front"),

        # 5. Goddess Pose (Front View)
        # Thighs parallel to floor (90° bend).
        ("Goddess Pose", "front_knee",  85, 110, None, "front"),
        ("Goddess Pose", "back_knee",   85, 110, None, "front"),
        ("Goddess Pose", "shoulders",  None, None, "horizontal", "front"),

        # 6. Triangle Pose (Side View)
        # Both legs are straight.
        ("Triangle Pose", "front_knee",  170, 180, None, "side"),
        # Torso is tilted; hip angle to front leg is roughly 80-100°.
        ("Triangle Pose", "hip",         80, 110, None, "side"),
    ]

    cursor.execute("DELETE FROM pose_rules;")

    for rule in rules:
        cursor.execute("""
            INSERT INTO pose_rules 
                (pose_name, joint_name, min_angle, max_angle, alignment_type, camera_view)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, rule)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Database seeded with {len(rules)} realistic posture rules!")

if __name__ == "__main__":
    seed_pose_rules()