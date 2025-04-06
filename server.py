import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import json
import eventlet
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://horizon-dash.vercel.app"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Prevent TensorFlow from using all memory
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
# Load the trained model
model = tf.keras.models.load_model("gesture_model.h5")
GESTURES = ["handOpen", "ThumbsUp", "peace", "fuck", "handClose", "rock", "left", "right"]

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_frame(frame):
    """ Preprocess the image for gesture recognition """
    # Resize the frame to a standard size
    frame = cv2.resize(frame, (640, 480))  

    # Convert to RGB (Mediapipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Increase brightness if needed
    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=30)

    results = hands.process(frame_rgb)
    
    
    if not results.multi_hand_landmarks:
        print("🔴 No hand detected")
        return "unknown"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            keypoints = np.array(keypoints).reshape(1, -1)

            print("🔵 Keypoints Shape:", keypoints.shape)  # Debugging
            prediction = model.predict(keypoints)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            print(f"🔵 Prediction: {prediction}, Class ID: {class_id}, Confidence: {confidence}")  # Debugging

            if confidence > 0.7:  # Confidence threshold
                print(GESTURES[class_id])
                return GESTURES[class_id]

    return "unknown"

@socketio.on("image")
def handle_frame(data):
    """ Receive and process image from the web client """
    try:
        print("🔵 Received data length:", len(data))

        image_data = data.split(",")[1]  # Remove "data:image/png;base64,"
        decoded_image = base64.b64decode(image_data)

        # Write to a temporary file (debugging)
        with open("debug_image.png", "wb") as f:
            f.write(decoded_image)

        # Convert to a PIL image
        image = Image.open(BytesIO(decoded_image))
        image = image.convert("RGB")  # Ensure it's in RGB format

        # Convert to OpenCV format
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR

        # Recognize gesture
        gesture = preprocess_frame(frame)

        # Send recognized gesture back
        socketio.emit("gesture", {"gesture": gesture})

    except Exception as e:
        print(f"❌ Error processing frame: {str(e)}")
        socketio.emit("error", {"message": "Failed to process image"})


@app.route("/")
def index():
    return "Gesture Recognition Server Running"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)
