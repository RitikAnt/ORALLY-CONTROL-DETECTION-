import threading
import queue
import time
import cv2
import mediapipe as mp
import pvporcupine
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import pyttsx3

ACCESS_KEY = "qfbN/nBItqLjUF6F8oNAhAJ6M0g5MW8IP5m6f33u5s4YlhQxAl+X9w=="
WAKE_WORD_PATH = "D:\\camera_voice_assistant\\model1\\titans_en_windows_v3_0_0.ppn"
VOSK_MODEL_PATH = "D:\\camera_voice_assistant\\model1"

porcupine = None
porcupine_audio_queue = queue.Queue()
wake_word_detected = threading.Event()
tts_engine = pyttsx3.init()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

BODY_PARTS = {
    "eye": [1, 2, 3, 4, 5, 6, 7, 8],
    "eyes": [1, 2, 3, 4, 5, 6, 7, 8],
    "hand": [15, 16, 17, 18, 19, 20],
    "hands": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "left hand": [21, 22, 23, 24, 25, 26],
    "right hand": [15, 16, 17, 18, 19, 20],
    "face": list(range(0, 11)),
    "head": list(range(0, 11)),
    "left foot": [27, 28, 29, 30],
    "right foot": [31, 32, 33, 34],
    "foot": [27, 28, 29, 30, 31, 32, 33, 34],
    "left leg": [23, 25, 27, 29],
    "right leg": [24, 26, 28, 30],
    "left arm": [11, 13, 15],
    "right arm": [12, 14, 16],
    "torso": [11, 12, 23, 24],
}

COLORS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
}

highlighting_on = True
highlight_color = COLORS["yellow"]
zoom_level = 1.0
latest_commands = []
circle_radius = 15
draw_connections = True
paused = False
state_lock = threading.Lock()

def tts_speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def wake_word_callback(indata, frames, time_info, status):
    if status and str(status) != 'input overflow':
        print(f"Sounddevice status: {status}")
    pcm = np.frombuffer(indata, dtype=np.int16)
    porcupine_audio_queue.put(pcm)

def wake_word_thread():
    global porcupine
    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keyword_paths=[WAKE_WORD_PATH])
    print("Porcupine wake word engine started.")
    try:
        with sd.InputStream(samplerate=porcupine.sample_rate, blocksize=porcupine.frame_length,
                            channels=1, dtype='int16', callback=wake_word_callback):
            while True:
                if not porcupine_audio_queue.empty():
                    pcm = porcupine_audio_queue.get()
                    if len(pcm) != porcupine.frame_length:
                        continue
                    result = porcupine.process(pcm)
                    if result >= 0:
                        print("Wake word detected!")
                        tts_speak("Yes?")
                        wake_word_detected.set()
                else:
                    time.sleep(0.01)
    except Exception as e:
        print(f"Wake word thread error: {e}")

def speech_recognition_thread():
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    def callback(indata, frames, time_info, status):
        if status and str(status) != 'input overflow':
            print(status)
        if wake_word_detected.is_set():
            data = indata.tobytes()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(f"Recognized speech: {text}")
                    handle_command(text)
                    wake_word_detected.clear()

    try:
        with sd.InputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
            print("Speech recognition started...")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Speech recognition error: {e}")

def handle_command(text):
    global latest_commands, highlighting_on, highlight_color, zoom_level, circle_radius, draw_connections, paused
    text = text.lower()
    with state_lock:
        if "turn on highlighting" in text or "highlighting on" in text:
            highlighting_on = True
            tts_speak("Highlighting turned on.")
            print("Highlighting turned ON")
            return
        if "turn off highlighting" in text or "highlighting off" in text:
            highlighting_on = False
            tts_speak("Highlighting turned off.")
            print("Highlighting turned OFF")
            return
        if "turn on connections" in text or "show connections" in text:
            draw_connections = True
            tts_speak("Drawing pose connections turned on.")
            print("Draw connections ON")
            return
        if "turn off connections" in text or "hide connections" in text:
            draw_connections = False
            tts_speak("Drawing pose connections turned off.")
            print("Draw connections OFF")
            return
        if "change highlight color to" in text:
            for color in COLORS.keys():
                if color in text:
                    highlight_color = COLORS[color]
                    tts_speak(f"Highlight color changed to {color}.")
                    print(f"Highlight color changed to {color}")
                    return
            tts_speak("Color not recognized. Available colors are red, green, blue, yellow, cyan, magenta, white.")
            return
        if "zoom in" in text:
            zoom_level = min(2.0, zoom_level + 0.1)
            tts_speak(f"Zoomed in to {zoom_level:.1f} times.")
            print(f"Zoom level: {zoom_level}")
            return
        if "zoom out" in text:
            zoom_level = max(1.0, zoom_level - 0.1)
            tts_speak(f"Zoomed out to {zoom_level:.1f} times.")
            print(f"Zoom level: {zoom_level}")
            return
        if "increase highlight size" in text or "increase circle size" in text:
            circle_radius = min(50, circle_radius + 5)
            tts_speak(f"Highlight size increased to {circle_radius} pixels.")
            print(f"Circle radius: {circle_radius}")
            return
        if "decrease highlight size" in text or "decrease circle size" in text:
            circle_radius = max(5, circle_radius - 5)
            tts_speak(f"Highlight size decreased to {circle_radius} pixels.")
            print(f"Circle radius: {circle_radius}")
            return
        if "pause camera" in text or "pause video" in text:
            paused = True
            tts_speak("Camera feed paused.")
            print("Camera feed paused")
            return
        if "resume camera" in text or "resume video" in text:
            paused = False
            tts_speak("Camera feed resumed.")
            print("Camera feed resumed")
            return
        if "take a screenshot" in text or "picture" in text:
            global last_frame
            if last_frame is not None:
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, last_frame)
                tts_speak(f"Screenshot saved as {filename}.")
                print(f"Screenshot saved: {filename}")
            else:
                tts_speak("No frame available to take a screenshot.")
            return
        if "help" in text:
            help_text = (
                "You can say commands like: focus on eyes and hands, "
                "turn off highlighting, change highlight color to red, "
                "zoom in, zoom out, increase highlight size, pause camera, "
                "take a screenshot, or help."
            )
            tts_speak(help_text)
            print(help_text)
        return
    parts = []
    for part in BODY_PARTS.keys():
        if part in text:
            parts.append(part)
    if parts:
        latest_commands = parts
        tts_speak(f"Focusing on {', '.join(parts)}")
        print(f"Focusing on {parts}")
    else:
        tts_speak("Command not recognized. Say help for available commands.")
        print("Unknown command")

last_frame = None

def main_camera_loop():
    global last_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            if paused:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            last_frame = frame.copy()
            if zoom_level != 1.0:
                height, width = frame.shape[:2]
                new_w = int(width / zoom_level)
                new_h = int(height / zoom_level)
                x1 = (width - new_w) // 2
                y1 = (height - new_h) // 2
                frame = frame[y1:y1+new_h, x1:x1+new_w]
                frame = cv2.resize(frame, (width, height))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks and highlighting_on:
                for part in latest_commands:
                    indices = BODY_PARTS.get(part, [])
                    for idx in indices:
                        landmark = results.pose_landmarks.landmark[idx]
                        h, w = frame.shape[:2]
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        if 0 <= cx < w and 0 <= cy < h:
                            cv2.circle(frame, (cx, cy), circle_radius, highlight_color, -1)
            if results.pose_landmarks and draw_connections:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Voice-Controlled Body Part Highlighter", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    wake_word_thread_obj = threading.Thread(target=wake_word_thread, daemon=True)
    speech_recog_thread_obj = threading.Thread(target=speech_recognition_thread, daemon=True)
    wake_word_thread_obj.start()
    speech_recog_thread_obj.start()
    main_camera_loop()
