import os
import sys
import argparse
from yolo import YOLO, detect_video
import cv2
import os
import matplotlib.pyplot as plt
import pytesseract as py
import pandas as pd
import numpy as np
from pathlib import Path

YOLO_DIR = 'DLCV/Detection/yolo/keras-yolo3/'

IMAGE_DIR = '/image'
CROP_DIR = '/image/Crop'

callnum_yolo = YOLO(model_path=os.path.join(YOLO_DIR,'snapshots/000/call number_final.h5'),
            anchors_path=os.path.join(YOLO_DIR, 'model_data/tiny_yolo_anchors.txt'),
            classes_path=os.path.join(YOLO_DIR, 'model_data/callnum_class.txt'))