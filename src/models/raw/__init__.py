# Raw-signal multi-task architectures (Phase 2).
# Input: (B, C=6, T=200)
from src.models.raw.cnn1d_raw import CNN1DRawMultiTask
from src.models.raw.lstm_raw import LSTMRawMultiTask
from src.models.raw.cnn_lstm_raw import CNNLSTMRawMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask

__all__ = [
    'CNN1DRawMultiTask',
    'LSTMRawMultiTask',
    'CNNLSTMRawMultiTask',
    'TCNRawMultiTask',
]
