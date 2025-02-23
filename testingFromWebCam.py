import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torchvision.models.video as video_models
import tensorflow as tf

def load_c3d_model():
    model = video_models.r3d_18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def display_video_with_predictions_from_webcam(model, tf_model, fps, clip_length=16):
    cap = cv2.VideoCapture(0)  # Use the default webcam
    frames_buffer = []  # Buffer to store frames for each clip

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        frame_tensor = transform(frame_pil)
        frames_buffer.append(frame_tensor)

        # Process a segment when enough frames are collected
        if len(frames_buffer) >= clip_length:
            # Create a segment tensor
            segment_tensor = torch.stack(frames_buffer[-clip_length:])  # Use the last `clip_length` frames
            segment_tensor = segment_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # Rearrange dimensions for model input

            # Extract features using the PyTorch model
            with torch.no_grad():
                if torch.cuda.is_available():
                    segment_tensor = segment_tensor.cuda()
                feature = model(segment_tensor).flatten(start_dim=1).cpu().numpy()

            # Apply padding to match TensorFlow model input size
            target_length = 4096
            if feature.shape[1] < target_length:
                padding = np.zeros((feature.shape[0], target_length - feature.shape[1]))
                feature = np.hstack((feature, padding))

            # Perform prediction using the TensorFlow model
            prediction = tf_model.predict(feature).flatten()
            label = "Anomaly" if prediction[0] > 0.5 else "Normal"
            color = (0, 0, 255) if label == "Anomaly" else (0, 255, 0)

            # Display the label and prediction on the video frame
            cv2.putText(frame, f"{label}: {prediction[0]:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the video frame
        cv2.imshow("Webcam Prediction", frame)

        # Control video playback speed based on FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load mô hình và sử dụng webcam
c3d_model = load_c3d_model()

model_path = "anomaly_detection_model3.h5"
tf_model = tf.keras.models.load_model(model_path)

# Dùng webcam để test
display_video_with_predictions_from_webcam(c3d_model, tf_model, fps=60)
