"""
A basic python file to store some variables that are used as parameters in fps_estimator.py
"""

from pathlib import Path

device = 'cuda'  # device selection for predictions cuda/cpu
model_dir = Path('F:\CV_FaceDetection\Models\FR\yolov8x-face.pt') # absolute path of the model being used for predictions
download_models = False  # whether to download pretrained model from gdrive or not, set False after the initial run


