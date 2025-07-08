import cv2
from keras.models import model_from_json
import numpy as np
import pywhatkit
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# =================== Load Emotion Model ===================
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())
model.load_weights("emotiondetector.h5")

# =================== Globals & Face Detection ===================
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

cap = None
running = False
last_emotion = None

# =================== GUI SETUP ===================
root = tk.Tk()
root.title("🎭 Emotion Music Player")
root.geometry("700x600")
root.resizable(False, False)

# Inputs
tk.Label(root, text="Language:", font=("Arial", 12)).pack(pady=5)
language_entry = tk.Entry(root, font=("Arial", 12), width=30)
language_entry.pack()

tk.Label(root, text="Singer Name:", font=("Arial", 12)).pack(pady=5)
singer_entry = tk.Entry(root, font=("Arial", 12), width=30)
singer_entry.pack()

# Video Frame
video_label = tk.Label(root)
video_label.pack(pady=10)

# =================== Emotion Detection Functions ===================
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def play_song_on_youtube(emotion, language, singer):
    query = f"{emotion} mood {language} song by {singer}"
    print(f"🎵 YouTube Search: {query}")
    try:
        pywhatkit.playonyt(query)
    except:
        print("❌ Could not play song.")

def update_frame():
    global cap, running, last_emotion

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_img = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img)
        pred = model.predict(img)
        emotion = labels[np.argmax(pred)]

        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        if emotion != last_emotion:
            last_emotion = emotion
            play_song_on_youtube(emotion, language_entry.get(), singer_entry.get())
            break  # avoid detecting multiple emotions in one frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule next frame update
    video_label.after(20, update_frame)

def start_detection():
    global cap, running, last_emotion
    lang = language_entry.get()
    singer = singer_entry.get()

    if not lang or not singer:
        messagebox.showerror("Input Required", "Please enter both language and singer name.")
        return

    cap = cv2.VideoCapture(0)
    last_emotion = None
    running = True
    update_frame()

def stop_detection():
    global cap, running
    running = False
    if cap:
        cap.release()
    video_label.configure(image="")

# =================== Buttons ===================
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="🎬 Start Detection", font=("Arial", 12),
                      bg="#4CAF50", fg="white", command=start_detection)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = tk.Button(btn_frame, text="🛑 Stop Detection", font=("Arial", 12),
                     bg="#f44336", fg="white", command=stop_detection)
stop_btn.grid(row=0, column=1, padx=10)

# Run GUI
root.mainloop()
