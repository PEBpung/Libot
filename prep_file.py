sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0

for i in range(1, len(crop_list)):
    crop_img = cv2.imread(CROP_DIR + str(i) + ".jpg")
    # 바운딩 박스 영역을 Gray로 변환
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    # rr-crop 과정을 거침
    height, width = gray.shape
    crop = gray[int( height / 4): height, 0 : int(15 * width / 16)].copy()
    # Sharpening을 사용하여 날카롭게 변환
    dst = cv2.filter2D(crop, -1, sharpening_2)
    # threshold the image using Otsus method to preprocess for tesseract
    thresh = cv2.adaptiveThreshold(dst,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    #perfrom bitwise not to flip image to black text on white background
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # 침식 연산 적용 ---②
    morph = cv2.erode(thresh, k)
    morph = cv2.dilate(morph, k)

    roi = cv2.bitwise_not(morph)
    #saving the roi regions
    cv2.imwrite(PREP_WAY1_v4_DIR + str(i) + ".jpg", roi)