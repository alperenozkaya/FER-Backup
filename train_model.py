"""
Ultralytics YOLOv8 - example of training a model using a pretrained model
"""

from ultralytics import YOLO


def main():
    model = YOLO('F:/CV_FaceDetection/runs/detect/train30/weights/best.pt')  # pretrained model is loaded using the absolute path

    # adjust the parameters to configure the training process
    model.train(data='FER_conf.yaml', epochs=100, imgsz=640, batch=8)


if __name__ == '__main__':
    main()