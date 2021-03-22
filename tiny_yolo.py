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
import tensorflow as tf
from PIL import Image
import time

print('tensorflow@@@@@@@@',tf.__version__)
print()
IMAGE_DIR = 'image'
CROP_DIR = 'image/Crop'

callnum_yolo = YOLO(model_path='keras-yolo3/model/call_number_final.h5',
            anchors_path='keras-yolo3/utils/tiny_yolo_anchors.txt',
            classes_path='keras-yolo3/utils/callnum_class.txt')

def ocr_to_crop_image(img, dir):
    CROP_DIR = os.path.join(dir, 'Crop')

    Width = img.shape[1]
    Height = img.shape[0]
    scale = 0.00392
    classes = None

    classes_path='keras-yolo3/utils/callnum_class.txt'
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    img = Image.fromarray(img)
    tmp_img = np.asarray(img.copy())
    outs_img, outs_box = callnum_yolo.detect_image(img)

    A = []
    for i, box in enumerate(outs_box):
        crop_img = tmp_img.copy()[box[0]:box[2], box[1]:box[3]]
        cv2.imwrite(CROP_DIR + "/crop__" + str(i) + ".jpg",crop_img)
        # 바운딩 박스 영역을 Gray로 변환
        gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        # 더 잘 인식되기 위해 3배로 Resize 시킴
        gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        # GaussianBlur를 사용하여 부드럽게 변환
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        # threshold the image using Otsus method to preprocess for tesseract
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(thresh)
        #saving the roi regions
        cv2.imwrite(CROP_DIR + "/roi__" + str(i) + ".jpg",roi)
        #passing to tesseract
        c = py.image_to_string(roi, lang='kor', config='--oem 1 --psm 11')
        A.append(c)

    B =[]
    N = len(A)
    for i in range(N):
        a = A[i].split('\n')
        a = list(filter(None, a))
        a.remove('\x0c')
        B.append(a[:3])

    return B

def detect_video_yolo(model, input_path, output_path=""):
    
    start = time.time()
    cap = cv2.VideoCapture(input_path)
    
    #codec = cv2.VideoWriter_fourcc(*'DIVX')
    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size)
    
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 Frame 갯수:', frame_cnt, '원본 영상 FPS:',vid_fps, '원본 Frame 크기:', vid_size)
    index = 0
    while True:
        hasFrame, image_frame = cap.read()
        if not hasFrame:
            print('프레임이 없거나 종료 되었습니다.')
            break
        start = time.time()
        # PIL Package를 내부에서 사용하므로 cv2에서 읽은 image_frame array를 다시 PIL의 Image형태로 변환해야 함.  
        image = Image.fromarray(image_frame)
        # 아래는 인자로 입력된 yolo객체의 detect_image()로 변환한다.
        detected_image, all = model.detect_image(image)
        # cv2의 video writer로 출력하기 위해 다시 PIL의 Image형태를 array형태로 변환 
        result = np.asarray(detected_image)
        index +=1
        print('#### frame:{0} 이미지 처리시간:{1}'.format(index, round(time.time()-start,3)))
        
        vid_writer.write(result)
    
    vid_writer.release()
    cap.release()
    print('### Video Detect 총 수행시간:', round(time.time()-start, 5))


VIDEO_DIR = 'video/'

detect_video_yolo(callnum_yolo, os.path.join(VIDEO_DIR, 'call_number_0.mp4'), 
                  os.path.join(VIDEO_DIR, 'output/call_number_0_yolo11.avi'))