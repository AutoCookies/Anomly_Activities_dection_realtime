import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torchvision.models.video as video_models
import tensorflow as tf

# Load mô hình C3D hoặc ResNet3D
def load_c3d_model():
    model = video_models.r3d_18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def video_to_segments(video_path, num_segments=32, clip_length=16):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        # Áp dụng các transformation
        frame_tensor = transform(frame_pil)
        frames.append(frame_tensor)

    cap.release()

    # Chia video thành segments
    segment_features = []
    segment_indices = []  # Lưu chỉ số frames cho từng segment
    for i in range(num_segments):
        start_idx = i * (total_frames // num_segments)
        segment_frames = frames[start_idx:start_idx + clip_length]
        if len(segment_frames) < clip_length:
            continue
        segment_tensor = torch.stack(segment_frames)
        segment_tensor = segment_tensor.permute(1, 0, 2, 3)
        segment_features.append(segment_tensor.unsqueeze(0))
        segment_indices.append((start_idx, start_idx + clip_length))

    return segment_features, segment_indices, fps

# Trích xuất đặc trưng từ một video
def extract_features_from_video(video_path, model, target_length=4096):
    segments, segment_indices, fps = video_to_segments(video_path)
    video_features = []

    for segment in segments:
        with torch.no_grad():
            # Di chuyển segment sang GPU nếu cần
            if torch.cuda.is_available():
                segment = segment.cuda()

            feature = model(segment).flatten(start_dim=1)
        video_features.append(feature.cpu().numpy())

    # Kết hợp tất cả đặc trưng và padding
    video_features = np.vstack(video_features)
    if video_features.shape[1] < target_length:
        padding = np.zeros((video_features.shape[0], target_length - video_features.shape[1]))
        video_features = np.hstack((video_features, padding))

    return video_features, segment_indices, fps

# Chiếu video và hiển thị dự đoán với thông tin Segment và Frame
def display_video_with_predictions(video_path, model, predictions, segment_indices, fps):
    cap = cv2.VideoCapture(video_path)
    current_segment = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Chỉ số frame hiện tại

        # Hiển thị dự đoán hiện tại và thông tin segment
        if current_segment < len(segment_indices):
            start_idx, end_idx = segment_indices[current_segment]
            if start_idx <= frame_idx < end_idx:
                label = "Anomaly" if predictions[current_segment] > 0.5 else "Normal"
                color = (0, 0, 255) if label == "Anomaly" else (0, 255, 0)

                # Tính chỉ số frame trong segment
                frame_in_segment = frame_idx - start_idx + 1

                # Vẽ label và thông tin segment lên frame
                cv2.putText(frame, f"{label}: {predictions[current_segment]:.2f}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Segment: {current_segment + 1}/{len(segment_indices)}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame in Segment: {frame_in_segment}/{end_idx - start_idx}", 
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Di chuyển sang segment tiếp theo nếu hết frame hiện tại
                if frame_idx == end_idx - 1:
                    current_segment += 1

        # Hiển thị frame
        cv2.imshow("Video Prediction", frame)

        # Điều chỉnh tốc độ phát dựa trên FPS
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load mô hình và trích xuất đặc trưng từ video kiểm tra
c3d_model = load_c3d_model()

# Đường dẫn video cần kiểm tra
test_video_path = "Data\Anomly_Videos\Assault023_x264.mp4"
test_features, segment_indices, fps = extract_features_from_video(test_video_path, c3d_model)

# Load mô hình TensorFlow đã huấn luyện
model_path = "anomaly_detection_model3.h5"
tf_model = tf.keras.models.load_model(model_path)

# Dự đoán
predictions = tf_model.predict(test_features).flatten()

# Hiển thị video với dự đoán
display_video_with_predictions(test_video_path, tf_model, predictions, segment_indices, fps)