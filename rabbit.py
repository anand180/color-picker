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
import json
from firebase import firebase
from firebase import firebase

firebase = firebase.FirebaseApplication('https://stylekick-colors.firebaseio.com/', None)


def read_url(url):
    req = urllib.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr,-1)
    r = 100.0 / img.shape[1]
    dim = (100, int(img.shape[0] * r))
    small = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    small = cv2.medianBlur(small,3)

    return small

def remove_skin(img):
    im_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    mask = cv2.bitwise_not(skin_ycrcb)
    # pdb.set_trace()
    new_img = np.zeros((img.shape[0],img.shape[1],4), np.uint8)
    return skin_ycrcb

def remove_bg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fg = cv2.erode(thresh,None,iterations = 2)
    bgt = cv2.dilate(thresh,None,iterations = 3)
    ret,bg = cv2.threshold(bgt,1,128,1)
    marker = cv2.add(fg,bg)
    marker32 = np.int32(marker)
    cv2.watershed(img,marker32)
    m = cv2.convertScaleAbs(marker32)
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # res = cv2.bitwise_and(img,img,mask = thresh)

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

    points = get_points(colors)
    clusters = kmeans(points, 3, 1)
    size = len(colors)/100
    sizes = [len(clusters[0].points)/size, len(clusters[1].points)/size, len(clusters[2].points)/size]
    rgbs = [map(int, c.center.coords) for c in clusters]
    print 'percentage: ' + str(sizes)
    print 'Color(org): ' + str(map(rtoh, rgbs))
    for x in range(3):
        save_color(sizes[x], map(rtoh, rgbs)[x])
    return rgbs[sizes.index(max(sizes))]

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
    max_p = img.shape[0] * img.shape[1]

    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            if cnoskin[r][c] == cbg[r][c] == 0:
                colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    if len(colors) < max_p/10:
        colors = []
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cnoskin[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    elif len(colors) < max_p/10:
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cbg[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    else:
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    return colors

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def get_color_name(p):
    platter = {}
    platter['#eeeee3'] = 'beige'
    platter['#2626d9'] = 'blue'
    platter['#864949'] = 'brown'
    platter['#d9bb26'] = 'gold'
    platter['#136d13'] = 'green'
    platter['#13136d'] = 'navy'
    platter['#f6c9d2'] = 'pink'
    platter['#d92626'] = 'red'
    platter['#c0c0c0'] = 'silver'
    platter['#d99726'] = 'orange'
    platter['#6d136d'] = 'purple'
    platter['#26d9d9'] = 'aqua'
    platter['#d9d626'] = 'yellow'
    platter['#808080'] = 'grey'
    platter['#000000'] = 'black'
    platter['#ffffff'] = 'white'
    rgbs = [(246, 201, 210), (109, 19, 109), (128, 128, 128), (217, 151, 38), (0, 0, 0), (19, 19, 109), (38, 217, 217), (192, 192, 192), (19, 109, 19), (217, 214, 38), (238, 238, 227), (217, 187, 38), (217, 38, 38), (38, 38, 217), (134, 73, 73), (255, 255, 255)]
    distances = []
    for c in rgbs:
        distances.append(int(sqrt(sum([(c[i] - p[i]) ** 2 for i in range(3)]))))
    # print distances
    # print min(distances)
    # print distances.index(min(distances))
    print platter[rgb_to_hex(rgbs[distances.index(min(distances))])]

def get_d_color(url):
    img = read_url(url)
    noskin = remove_skin(img)
    bg = remove_bg(img)
    cbg = crop(bg)
    cnoskin = crop(noskin)
    img = crop(img)
    cv2.imwrite('cbg.png', cbg)
    cv2.imwrite('cnoskin.png', cnoskin)
    cv2.imwrite('img.png', img)

    try:
        colors = get_colors(img, cbg, cnoskin)
        if len(colors) < 1000:
            print 'Try: skin covers most of picture'
            colors = get_colors(img, cbg, [])
    except Exception, e:
        print 'Try(e): it without skin removal.'
        colors = get_colors(img, cbg, [])

    get_color_name(colorz(colors))

def get_styles(url):
    req = urllib.urlopen(url).read()
    result = json.loads(req)
    for x in range(30):
        styles = result['styles'][x]['product']['styles']
        for style in styles:
            print style['large_grid_image_url']
            get_d_color(style['large_grid_image_url'])


def save_color(per, v):
    result = firebase.post('/colors/' + str(per), v)
    print result


if __name__ == "__main__":
    # for page in range(1,100):
    #     url = 'http://www.stylekick.com/api/v1/styles?page='
    #     get_styles(url + str(page))
    get_d_color('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/9420/large_grid_4377_white')





