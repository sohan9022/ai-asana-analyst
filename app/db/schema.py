from app.db.connection import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     SERIAL PRIMARY KEY,
            username    VARCHAR(100) UNIQUE NOT NULL,
            email       VARCHAR(150) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pose_rules (
            pose_id        SERIAL PRIMARY KEY,
            pose_name      VARCHAR(100) NOT NULL,
            joint_name     VARCHAR(100) NOT NULL,
            min_angle      FLOAT ,
            max_angle      FLOAT ,
            alignment_type VARCHAR(50),
            camera_view    VARCHAR(50)
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_reports (
            report_id           SERIAL PRIMARY KEY,
            user_id             INT REFERENCES users(user_id),
            pose_name           VARCHAR(100),
            timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            detected_angles     JSON,
            violations          TEXT[],
            undetectable_frames INT DEFAULT 0,
            score               FLOAT
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("All tables created successfully!")

if __name__ == "__main__":
    create_tables()