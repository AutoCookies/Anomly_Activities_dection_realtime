from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Reshape, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2

class AnomalyDetectionModel:
    def __init__(self):
        self.model = Sequential([
            Dense(1024, input_dim=4096, activation='relu', kernel_regularizer=l2(0.0010000000474974513)),
            Dropout(0.6),
            Dense(512, activation='relu', kernel_regularizer=l2(0.0010000000474974513)),
            Dropout(0.6),
            Dense(32, activation='linear', kernel_regularizer=l2(0.0010000000474974513)),
            Dropout(0.6),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0010000000474974513) )
        ])

    def get_model(self):
        return self.model