import numpy as np
import matplotlib.pyplot as plt
import cv2


def max_6(coutours):
    S = np.ones(len(coutours))
    S_xywh = np.full((len(coutours), 4), 0)
    Max_six = np.full((20, 4), 0)
    j = 0
    for c in coutours:
        x, y, w, h = cv2.boundingRect(c)
        S[j] = (w+h)*2
        S_xywh[j] = [x, y, w, h]
        j += 1
    S[0] = 1
    for i in range(6):
        S[np.where(S == max(S))] = 0
    m = 0

    for i in range(len(S)):
        if(S[i] == 0):
            Max_six[m] = S_xywh[i]
            m += 1
    return Max_six


for i in range(10):
    name_test = './DATA_TEST/' + str(i) + ".png"
    img_test = cv2.imread(name_test)
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    img_test = cv2.resize(src=img_test, dsize=(1200, 400))
    ret, thresh = cv2.threshold(img_test, 127, 255, 0)
    coutours, hierarphy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    MAX_S = np.array([0, 1, 2, 3, 4, 5])
    coutours = np.asarray(coutours)
    crop_xywh = max_6(coutours)
    place = 15
    for j in range(6):
        print('x,y,w,h:', crop_xywh[j])
        crop_img = img_test[crop_xywh[j][1] - place:crop_xywh[j][1] + crop_xywh[j]
                            [3] + place, crop_xywh[j][0] - place:crop_xywh[j][0]+crop_xywh[j][2]+place]
        crop_img = cv2.resize(src=crop_img, dsize=(28, 28))
        cv2.imwrite('./blog/%s_%s.png' % (i, j), crop_img)
        for i in range(len(crop_xywh)):
            cv2.rectangle(img_test, (crop_xywh[i][0], crop_xywh[i][1]), (
                crop_xywh[i][0]+crop_xywh[i][2], crop_xywh[i][1]+crop_xywh[i][3]), (0, 255, 0), 2)
    #     print('===================================')
    # cv2.drawContours(img_test, coutours, -1, (0, 255, 0), 1)
    # cv2.imshow('qwg', img_test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
