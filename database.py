import sqlite3
import json
from datetime import datetime

def init_database():
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    # First, drop the existing table to start fresh
    cursor.execute('DROP TABLE IF EXISTS predictions')
    
    # Create predictions table with ALL columns including is_mitigated
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_name TEXT NOT NULL,
            input_features TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            confidence REAL,
            is_adversarial BOOLEAN DEFAULT 0,
            is_mitigated BOOLEAN DEFAULT 0,
            mitigation_suggestion TEXT
        )
    ''')
    
    # Create model statistics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            total_predictions INTEGER DEFAULT 0,
            attack_detections INTEGER DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database recreated successfully with is_mitigated column")

def get_db_connection():
    conn = sqlite3.connect('ids_database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to verify the table structure
def verify_table_structure():
    conn = sqlite3.connect('ids_database.db')
    cursor = conn.cursor()
    
    # Get all column names
    cursor.execute("PRAGMA table_info(predictions)")
    columns = cursor.fetchall()
    
    print("📊 Current table structure:")
    print("Column Name | Data Type")
    print("-" * 30)
    for col in columns:
        print(f"{col[1]:<15} | {col[2]}")
    
    conn.close()
    return columns

# Run this to initialize and verify
if __name__ == "__main__":
    init_database()
    verify_table_structure()