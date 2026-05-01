import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data/memory.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS mastery (
            student_id TEXT,
            topic TEXT,
            score TEXT,
            tutor_note TEXT,
            student_summary TEXT,
            session_date TEXT,
            subject TEXT DEFAULT anatomy
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized")

def save_mastery(student_id, topic, score, tutor_note, student_summary, subject = "anatomy"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO mastery VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        topic,
        score,
        tutor_note,
        student_summary,
        datetime.now().isoformat(),
        subject
    ))
    conn.commit()
    conn.close()

def load_weak_spots(student_id, subject = "anatomy"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT topic, tutor_note FROM mastery
        WHERE student_id = ? AND score != 'strong' AND subject = ?
        ORDER BY session_date DESC
        LIMIT 5
    """, (student_id,))
    rows = c.fetchall()
    conn.close()
    return [f"{row[0]}: {row[1]}" for row in rows]

def get_student_summary(student_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT topic, score, student_summary, session_date FROM mastery
        WHERE student_id = ?
        ORDER BY session_date DESC
    """, (student_id,))
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    save_mastery(
        student_id="test_student",
        topic="brachial plexus",
        score="partial",
        tutor_note="Student understands origins but unclear on clinical implications",
        student_summary="You showed good understanding of spinal levels but struggled with clinical application."
    )
    print("Saved mastery record")
    
    weak_spots = load_weak_spots("test_student")
    print("Weak spots:", weak_spots)
    
    summary = get_student_summary("test_student")
    print("Full summary:")
    for row in summary:
        print(f"  Topic: {row[0]} | Score: {row[1]} | Date: {row[3]}")