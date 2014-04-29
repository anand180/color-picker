import sys, json, time, traceback
import cv2
# from skimage import io
import urllib
import numpy as np
import pdb


def read_url(url):
    req = urllib.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return cv2.imdecode(arr,-1)

def remove_skin(img):
    im_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    mask = cv2.bitwise_not(skin_ycrcb)
    # pdb.set_trace()

    new_img = np.zeros((img.shape[0],img.shape[1],4), np.uint8)

    # r = 0
    # for row in mask:
    #     c = 0
    #     for b_pixal in row:
    #         if b_pixal == 0:
    #             new_img[r][c] = [0, 0, 0, 0]
    #         else:
    #             new_img[r][c] = [img[r][c][0], img[r][c][1], img[r][c][2], 255]
    #         c += 1
    #     r += 1


    skin_ycrcb = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite('skin.png', skin_ycrcb)
    return skin_ycrcb
    # cv2.imshow('img', skin_ycrcb)

def remove_bg(img):
    cv2.imwrite('bg-org.png', img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('bg-gry.png', gray)

    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_TOZERO)
    cv2.imwrite('bg-pre.png', thresh)

    fg = cv2.erode(thresh,None,iterations = 2)
    bgt = cv2.dilate(thresh,None,iterations = 3)
    ret,bg = cv2.threshold(bgt,1,128,1)
    marker = cv2.add(fg,bg)
    marker32 = np.int32(marker)
    cv2.watershed(img,marker32)
    m = cv2.convertScaleAbs(marker32)
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    res = cv2.bitwise_and(img,img,mask = thresh)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # res = cv2.bitwise_and(img,img,mask = thresh)
    cv2.imwrite('bg.png', thresh)


if __name__ == "__main__":
    img = read_url('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/172280/rsa8335_neonyellow')
    # no_skin_img = remove_skin(img)
    remove_bg(img)
