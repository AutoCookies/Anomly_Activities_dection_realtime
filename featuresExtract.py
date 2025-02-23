import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torchvision.models.video as video_models

def load_c3d_model():
    model = video_models.r3d_18(pretrained=True)  # Load R3D từ torchvision
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp FC cuối cùng
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def video_to_segments(video_path, num_segments=32, clip_length=16):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Resize frames về (112, 112)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi frame sang RGB rồi về PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)  # Chuyển về PIL Image

        # Áp dụng các transformation
        frame_tensor = transform(frame_pil)
        frames.append(frame_tensor)

    cap.release()

    # Chia video thành segments
    segment_features = []
    for i in range(num_segments):
        start_idx = i * (total_frames // num_segments)
        segment_frames = frames[start_idx:start_idx + clip_length]
        if len(segment_frames) < clip_length:
            continue
        segment_tensor = torch.stack(segment_frames)
        segment_tensor = segment_tensor.permute(1, 0, 2, 3)
        segment_features.append(segment_tensor.unsqueeze(0))

    return segment_features

def extract_features_from_folder(folder_path, model, output_path):
    all_features = []
    video_names = []
    video_count = 0

    for video_file in tqdm(os.listdir(folder_path)):
        if video_count == 100:
            break
        
        if video_file.endswith(('.mp4', '.avi', '.mkv')):  # Các định dạng video hợp lệ
            video_path = os.path.join(folder_path, video_file)
            try:
                segments = video_to_segments(video_path)
                video_features = []

                for segment in segments:
                    with torch.no_grad():
                        # Di chuyển segment sang GPU nếu cần
                        if torch.cuda.is_available():
                            segment = segment.cuda()

                        feature = model(segment).flatten(start_dim=1)

                    video_features.append(feature.cpu().numpy())

                all_features.append(np.vstack(video_features))  # Nối các đặc trưng của các segment
                video_names.append(video_file)

            except ValueError as e:
                print(f"Warning: {e}")  # Bỏ qua video lỗi
        video_count += 1
    # Lưu đặc trưng vào file output
    np.save(output_path, {"features": all_features, "videos": video_names})
    print(f"Đã lưu đặc trưng vào: {output_path}")
    
# Load mô hình và di chuyển sang GPU nếu có
model = load_c3d_model()

# Đường dẫn thư mục chứa video
base_path = "Data"
anomaly_folder = os.path.join(base_path, "Anomly_Videos")
normal_folder = os.path.join(base_path, "Normal_Videos")

# Đường dẫn lưu file output
Feature_path = "Features"
os.makedirs(Feature_path, exist_ok=True)

anomaly_output = os.path.join(Feature_path, "anomaly_features2.npy")
normal_output = os.path.join(Feature_path, "normal_features2.npy")


# Trích xuất đặc trưng cho Anomaly Videos
print("Đang xử lý Anomaly Videos...")
extract_features_from_folder(anomaly_folder, model, anomaly_output)

# Trích xuất đặc trưng cho Normal Videos
print("Đang xử lý Normal Videos...")
extract_features_from_folder(normal_folder, model, normal_output)