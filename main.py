from fastapi import FastAPI, UploadFile, File, Form, Depends
import uvicorn
import numpy as np
import cv2
from datetime import datetime
import psycopg2
import json
from psycopg2.extras import RealDictCursor

app = FastAPI()

# Подключение к базе данных PostgreSQL
try:
    conn = psycopg2.connect(
        dbname="ingria_db", user="user", password="password", host="localhost", port="5432",
        cursor_factory=RealDictCursor
    )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sensor_type TEXT NOT NULL,
            data JSONB NOT NULL
        )
    ''')
    conn.commit()
except Exception as e:
    print(f"Database connection error: {e}")

@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Простейшая обработка изображения
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Сохранение воспоминания
        memory = {
            "image_analysis": "edges_detected",
            "shape": img.shape
        }
        cursor.execute(
            "INSERT INTO memories (sensor_type, data) VALUES (%s, %s)",
            ("camera", json.dumps(memory))
        )
        conn.commit()
        
        return {"message": "Image processed", "memory": memory}
    except Exception as e:
        return {"error": str(e)}

@app.post("/log_sound/")
async def log_sound(duration: float = Form(...)):
    try:
        memory = {
            "sound_detected": True,
            "duration": duration
        }
        cursor.execute(
            "INSERT INTO memories (sensor_type, data) VALUES (%s, %s)",
            ("microphone", json.dumps(memory))
        )
        conn.commit()
        
        return {"message": "Sound logged", "memory": memory}
    except Exception as e:
        return {"error": str(e)}

@app.get("/memories/")
def get_memories():
    try:
        cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT 10")
        records = cursor.fetchall()
        return {"recent_memories": records}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

