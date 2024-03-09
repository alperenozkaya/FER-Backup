"""
A modified program to estimate the average FPS for predictions on sample videos.
Sample code source: https://docs.ultralytics.com/guides/security-alarm-system/#email-received-sample
"""

import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import config
import gdown


class ObjectDetection:
    def __init__(self, capture_index, model_dir):
        # default parameters
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        self.model_name = model_dir.stem
        self.model = YOLO(model_dir)

        # device information
        self.device = config.device

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.fps = 0
        self.average_fps = 0
        self.fps_instances = []

    def predict(self, im0):
        results = self.model.predict(im0, device= config.device)
        return results

    # function to display fps
    def display_fps(self, im0):
        self.end_time = time()
        self.fps = 1 / (self.end_time - self.start_time)
        self.fps_instances.append(self.fps)
        text = f'FPS: {int(self.fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # function plot boundary boxes
    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im0, class_ids

    # write results into a text file in the desired format
    def write_results_to_file(self):
        with open("FPS_Results.txt", "a") as file:
            file.write(
                f"Device: {self.device} Model: {detector.model_name} Average FPS: {self.average_fps}\n")

    # main call function for video capturing
    def __call__(self):
        cap = cv2.VideoCapture('sample1.mp4')
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0

        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            if not ret:
                break  # break the loop if the video ends

            results = self.predict(im0)  # store the results of prediction
            im0, class_ids = self.plot_bboxes(results, im0)  # draw boundary boxes - get image result, and predicted classes

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == 27:  # break if 'Esc' key is pressed
                break
        self.average_fps = sum(self.fps_instances) / len(self.fps_instances)  # get average fps of predictions
        print(self.average_fps)
        self.write_results_to_file()

        cap.release()
        cv2.destroyAllWindows()


# a function to download pretrained YOLOv8-YOLOv8-FR-YOLOv8-FER models
def download_models_from_google_drive():
    datasets_drive_url = 'https://drive.google.com/drive/folders/108xWOWt8jbva-UDKn37RhyBDO4VEz_Je?usp=drive_link'
    if datasets_drive_url.split('/')[-1] == '?usp=drive_link':
        datasets_drive_url = datasets_drive_url.replace('?usp=drive_link', '')

    gdown.download_folder(datasets_drive_url)


if config.download_models:
    download_models_from_google_drive()

model_path = config.model_dir
model_name = model_path.stem  # extract the name using pathlib

detector = ObjectDetection(capture_index=0, model_dir=config.model_dir)
detector()