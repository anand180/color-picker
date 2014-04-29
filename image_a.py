import sys, json, time, traceback
import cv2
# from skimage import io
import urllib
import numpy as np
import pdb
from collections import namedtuple
from math import sqrt
import random
from PIL import Image

def read_url(url):
    req = urllib.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr,-1)
    r = 50.0 / img.shape[1]
    dim = (50, int(img.shape[0] * r))
    small = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    small = cv2.medianBlur(small,5)
    cv2.imwrite('small.png', small)
    return small

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

    cv2.imwrite('noskin.png', skin_ycrcb)

    # skin_ycrcb = cv2.bitwise_and(img, img, mask=mask)
    return skin_ycrcb
    # cv2.imshow('img', skin_ycrcb)

def remove_bg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fg = cv2.erode(thresh,None,iterations = 2)
    bgt = cv2.dilate(thresh,None,iterations = 3)
    ret,bg = cv2.threshold(bgt,1,128,1)
    cv2.imwrite('bg-1.png', bg)

    marker = cv2.add(fg,bg)
    marker32 = np.int32(marker)
    cv2.watershed(img,marker32)
    m = cv2.convertScaleAbs(marker32)
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('bg-2.png', thresh)

    # res = cv2.bitwise_and(img,img,mask = thresh)
    # cv2.imwrite('bg.png', res)

    return thresh

def crop(img):
    shape = img.shape
    crop = img[shape[0]/4:shape[0]/4*3, shape[1]/4:shape[1]/4*3]
    return crop

Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))

def get_points(colors):
    points = []
    # pdb.set_trace()

    for count, color in enumerate(colors):
        points.append(Point(color, 3, count))
    pdb.set_trace()

    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(colors, n=3):
    # img = Image.open(img)
    # w, h = img.size

    points = get_points(colors)
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([
        (p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)
    ]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]

    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center, new.center))

        if diff < min_diff:
            break

    return clusters

def get_colors(img, cnoskin, cbg):
    colors = []
    for r in xrange(0,img.shape[0]):
        for c in xrange(0,img.shape[1]):
            if cnoskin[r][c] == cbg[r][c] == 255:
                colors.append((img[r][c][0],img[r][c][1],img[r][c][2]))
    return colors

if __name__ == "__main__":
    img = read_url('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/12356/rsacs300_salmon')
    noskin = remove_skin(img)
    bg = remove_bg(img)
    cbg = crop(bg)
    cnoskin = crop(noskin)
    img = crop(img)
    cv2.imwrite('cbg.png', cbg)
    cv2.imwrite('cnoskin.png', cnoskin)
    cv2.imwrite('img.png', img)

    colors = get_colors(img, cnoskin, cbg)

    print colorz(colors)

