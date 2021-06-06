import cv2
import os
import matplotlib.pyplot as plt
import pytesseract as py
import pandas as pd
import numpy as np
import time 
import re 

HOME_DIR = '/content/drive/MyDrive/Ocr_Test/'
DATA_DIR = os.path.join(HOME_DIR, 'data/crop_to_prep/')

OCR_TXT_DIR = os.path.join(DATA_DIR, 'ocr_to_txt/')
PREP_DIR = os.path.join('')

def saveFile(cur_text, cur_txt_num):
    if (cur % cycle == 0 or i == (len(file_list) - 1) ):
        cur_text = {key: value for key, value in cur_text.items() if value > 2}

        for txt in cur_text.keys():
            if txt in send_text:
                continue
            else:
                send_text.append(txt)

        f = open(OCR_TXT_DIR + 'send_file.txt', 'w')
        f.write(','.join(send_text))
        f.close()

        print('------ {} 번째 txt 파일 저장 완료-----'.format(cur_txt_num))
        print(send_text)
        print()

def inference_ocr(PREP_DIR, cycle):
    '''
    PREP_DIR : 전처리 파일의 위치
    cycle : txt 전송 주기
    '''
    start = 1
    cur_num = 1
    comp_num = 1

    cur_text = {}
    all_time = 0
    cur_txt_num = 1

    send_text = []
    detect_ocr_cnt = 0

    while True:
        file_list = os.listdir(PREP_DIR)

        for i in range(cur_num, len(file_list)):
            st_sec = time.time()
            
            cur_num += 1

            # 폴더에 들어오는 파일이 없으면 무한 대기.
            if (i > len(file_list)):
                print('파일 입력 대기중.....')
                continue

            if (i % 100 == 0):
                print('------ {}번째 파일 수행 완료-----'.format(i))

            img = PREP_DIR + str(i) + '.jpg'

            # OCR 실행 
            ocr_str = py.image_to_string(img, lang='eng', config='--oem 1 --psm 11')

            # 검출된 문자 유무 체크
            s = ocr_str.split()
            if not s: continue

            # ocr 전처리 : 숫자를 제외한 문자 제거.
            p = re.compile('[0-9]')
            num = ''.join(p.findall(s[0]))
            if(len(num) < 4): continue

            # 제대로 검출된 문자 카운트.
            detect_ocr_cnt += 1

            # 검출된 청구기호 dict에 저장.
            if not num in cur_text.keys():
                cur_text[num] = 1
            else:
                cur_text[num] += 1
                
            end_sec = time.time()
            print('경과시간 : {}'.format(end_sec - st_sec))
            print(cur_text)
            print()
                
            cur += 1
            start += 10

            # 청구기호 검출 파일 저장.
            saveFile(cur_text, cur_txt_num)
            
            cur_txt_num += 1

    print('------------------------')
    print('제대로 검출된 숫자 : {}'.format(detect_ocr_cnt))
    print('검출 확률 : {}% '.format(round(detect_ocr_cnt / len(file_list), 2) * 100))