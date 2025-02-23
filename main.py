import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from Model import AnomalyDetectionModel

def pad_features(features, target_length):
    num_samples, current_length = features.shape
    if current_length < target_length:
        padding = np.zeros((num_samples, target_length - current_length))
        return np.hstack((features, padding))
    return features

# Load data
anomaly_data = np.load("Features//anomaly_features2.npy", allow_pickle=True).item()
normal_data = np.load("Features//normal_features2.npy", allow_pickle=True).item()

anomaly_features = np.vstack(anomaly_data["features"])
normal_features = np.vstack(normal_data["features"])

anomaly_labels = np.ones(len(anomaly_features))
normal_labels = np.zeros(len(normal_features))

features = np.concatenate((anomaly_features, normal_features), axis=0)
labels = np.concatenate((anomaly_labels, normal_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train = pad_features(X_train, 4096)
X_val = pad_features(X_val, 4096)
X_test = pad_features(X_test, 4096)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

model = AnomalyDetectionModel().get_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=64, callbacks=[early_stopping])

model_path = "anomaly_detection_model4.h5"
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"{model_path} already existed and has been removed.")
model.save(model_path)

print(f"Training complete. Model saved as {model_path}")

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs > 0.5
print("AUC-ROC:", roc_auc_score(y_test, y_pred_probs))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))