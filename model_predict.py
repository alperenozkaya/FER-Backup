from ultralytics import YOLO

model = YOLO('F:/CV_FaceDetection/runs/detect/train24/weights/best.pt')

results = model(source='selfie.jpg', show=True, conf=0.4, save=True, boxes=True, stream=False, show_labels=True)