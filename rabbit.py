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
    r = 100.0 / img.shape[1]
    dim = (100, int(img.shape[0] * r))
    small = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    small = cv2.medianBlur(small,3)
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

    for count, color in enumerate(colors):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(colors, n=3):
    # img = Image.open(img)
    # w, h = img.size

    points = get_points(colors)
    clusters = kmeans(points, 3, 1)
    size = len(colors)/100
    # pdb.set_trace()
    print size

    sizes = [len(clusters[0].points)/size, len(clusters[1].points)/size, len(clusters[2].points)/size]
    print sizes
    rgbs = [map(int, c.center.coords) for c in clusters]
    return rgbs[sizes.index(max(sizes))]

    # return rgbs
    # map(rtoh, rgbs)

def euclidean(p1, p2):
    return sqrt(sum([(p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)]))

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

def get_colors(img, cbg, cnoskin):
    colors = []
    if len(cnoskin) != 0:
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cnoskin[r][c] == cbg[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    else:
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cbg[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    return colors

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def get_color_name(p):
    # platter = {}
    # platter['#eeeee3'] = 'beige'
    # platter['#2626d9'] = 'blue'
    # platter['#864949'] = 'brown'
    # platter['#d9bb26'] = 'gold'
    # platter['#136d13'] = 'green'
    # platter['#13136d'] = 'navy'
    # platter['#f6c9d2'] = 'pink'
    # platter['#d92626'] = 'red'
    # platter['#c0c0c0'] = 'silver'
    # platter['#d99726'] = 'orange'
    # platter['#6d136d'] = 'purple'
    # platter['#26d9d9'] = 'aqua'
    # platter['#d9d626'] = 'yellow'
    # platter['#808080'] = 'grey'
    # platter['#000000'] = 'black'
    # platter['#ffffff'] = 'white'
    rgbs = [(0, 0, 0), (0, 0, 64), (0, 0, 128), (0, 0, 192), (0, 0, 256), (0, 64, 0), (0, 64, 64), (0, 64, 128), (0, 64, 192), (0, 64, 256), (0, 128, 0), (0, 128, 64), (0, 128, 128), (0, 128, 192), (0, 128, 256), (0, 192, 0), (0, 192, 64), (0, 192, 128), (0, 192, 192), (0, 192, 256), (0, 256, 0), (0, 256, 64), (0, 256, 128), (0, 256, 192), (0, 256, 256), (64, 0, 0), (64, 0, 64), (64, 0, 128), (64, 0, 192), (64, 0, 256), (64, 64, 0), (64, 64, 64), (64, 64, 128), (64, 64, 192), (64, 64, 256), (64, 128, 0), (64, 128, 64), (64, 128, 128), (64, 128, 192), (64, 128, 256), (64, 192, 0), (64, 192, 64), (64, 192, 128), (64, 192, 192), (64, 192, 256), (64, 256, 0), (64, 256, 64), (64, 256, 128), (64, 256, 192), (64, 256, 256), (128, 0, 0), (128, 0, 64), (128, 0, 128), (128, 0, 192), (128, 0, 256), (128, 64, 0), (128, 64, 64), (128, 64, 128), (128, 64, 192), (128, 64, 256), (128, 128, 0), (128, 128, 64), (128, 128, 128), (128, 128, 192), (128, 128, 256), (128, 192, 0), (128, 192, 64), (128, 192, 128), (128, 192, 192), (128, 192, 256), (128, 256, 0), (128, 256, 64), (128, 256, 128), (128, 256, 192), (128, 256, 256), (192, 0, 0), (192, 0, 64), (192, 0, 128), (192, 0, 192), (192, 0, 256), (192, 64, 0), (192, 64, 64), (192, 64, 128), (192, 64, 192), (192, 64, 256), (192, 128, 0), (192, 128, 64), (192, 128, 128), (192, 128, 192), (192, 128, 256), (192, 192, 0), (192, 192, 64), (192, 192, 128), (192, 192, 192), (192, 192, 256), (192, 256, 0), (192, 256, 64), (192, 256, 128), (192, 256, 192), (192, 256, 256), (256, 0, 0), (256, 0, 64), (256, 0, 128), (256, 0, 192), (256, 0, 256), (256, 64, 0), (256, 64, 64), (256, 64, 128), (256, 64, 192), (256, 64, 256), (256, 128, 0), (256, 128, 64), (256, 128, 128), (256, 128, 192), (256, 128, 256), (256, 192, 0), (256, 192, 64), (256, 192, 128), (256, 192, 192), (256, 192, 256), (256, 256, 0), (256, 256, 64), (256, 256, 128), (256, 256, 192), (256, 256, 256)]
    distances = []
    for c in rgbs:
        distances.append(int(sqrt(sum([(c[i] - p[i]) ** 2 for i in range(3)]))))
    print distances
    print min(distances)
    print distances.index(min(distances))
    print rgb_to_hex(rgbs[distances.index(min(distances))])

if __name__ == "__main__":
    img = read_url('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/89530/2468e13322add2774fc012e038a59dc2_best.jpg')
    noskin = remove_skin(img)
    bg = remove_bg(img)
    cbg = crop(bg)
    cnoskin = crop(noskin)
    img = crop(img)
    cv2.imwrite('cbg.png', cbg)
    cv2.imwrite('cnoskin.png', cnoskin)
    cv2.imwrite('img.png', img)

    # colors = get_colors(img, cbg, cnoskin)

    try:
        colors = get_colors(img, cbg, cnoskin)
        if len(colors) < 1000:
            colors = get_colors(img, cbg, [])
    except Exception, e:
        print 'i am in try'
        colors = get_colors(img, cbg, [])



    colors = get_colors(img, cnoskin, cbg)
    colorz(colors)
    get_color_name(colorz(colors))


