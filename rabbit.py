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
from colormath.color_objects import RGBColor

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

firebase = firebase.FirebaseApplication('https://stylekick-colors.firebaseio.com/', None)
platter = {}
platter['#fffff0'] = 'Ivory'
platter['#87ceeb'] = 'LightBlue'
platter['#8800CC'] = 'Purple'
platter['#808080'] = 'Gray'
platter['#008000'] = 'Green'
platter['#0000ff'] = 'Blue'
platter['#683300'] = 'Brown'
platter['#000000'] = 'Black'
platter['#efdeb0'] = 'Beige'
platter['#000080'] = 'Navy'
platter['#ffa500'] = 'Orange'
platter['#ff0000'] = 'Red'
platter['#ffffff'] = 'White'
platter['#00ffff'] = 'Aqua'
platter['#ffff00'] = 'Yellow'
platter['#9acd32'] = 'YellowGreen'
platter['#FF69B4'] = 'Pink'
platter['#5B0124'] = 'Red'
platter['#d6a789'] = 'SkinColor'



platter['#c8c8c3'] = 'Gray'
platter['#d3d1d4'] = 'Gray'
platter['#898b2f'] = 'Green'
platter['#1f2a33'] = 'Black'
platter['#e7e7f0'] = 'White'
platter['#d8ee99'] = 'YellowGreen'
platter['#d4efb6'] = 'YellowGreen'
platter['#b4f8c1'] = 'YellowGreen'
platter['#c29582'] = 'SkinColor'
platter['#d4a694'] = 'SkinColor'
platter['#d1ab99'] = 'SkinColor'
platter['#c88f6f'] = 'SkinColor'
platter['#f47f6e'] = 'Red'
platter['#e8a1b1'] = 'Pink'
platter['#415877'] = 'Navy'
platter['#dadbeb'] = 'Purple'
platter['#cfdee8'] = 'LightBlue'
platter['#bfb3a3'] = 'Beige'
platter['#1f61c3'] = 'Blue'
platter['#337074'] = 'Green'
platter['#028590'] = 'Green'
platter['#edd7d0'] = 'SkinColor'
platter['#7c8daa'] = 'Navy'
platter['#575a67'] = 'Gray'
platter['#6250a1'] = 'Purple'
platter['#9289a8'] = 'Purple'
platter['#6a96bf'] = 'LightBlue'

platter['#2f2e2a'] = 'Gray'
platter['#3b352c'] = 'Green'
platter['#4a4442'] = 'Black'
platter['#7e8c9b'] = 'blue'
platter['#181d29'] = 'Navy'
platter['#212b45'] = 'Navy'
platter['#242b40'] = 'Navy'
platter['#0b3336'] = 'Green'
platter['#5b5957'] = 'Gray'
platter['#6a84a4'] = 'LightBlue'
platter['#947f75'] = 'Beige'
platter['#969caa'] = 'Gray'
platter['#7689c3'] = 'LightBlue'
platter['#8193c0'] = 'LightBlue'
platter['#e6ded2'] = 'Beige'
platter['#f3efeb'] = 'Ivory'
platter['#231e1b'] = 'Black'
platter['#18171d'] = 'Black'
platter['#49474d'] = 'Gray'
platter['#cfced9'] = 'Gray'
platter['#d4cec9'] = 'Beige'
platter['#f0eff1'] = 'White'
platter['#3c3d4c'] = 'Navy'
platter['#6e636f'] = 'Purple'
platter['#6b5b3f'] = 'Brown'
platter['#9b6649'] = 'SkinColor'
platter['#27716d'] = 'Green'
platter['#123279'] = 'blue'
platter['#574332'] = 'Brown'
platter['#dec6c1'] = 'Pink'
platter['#e8a79d'] = 'Pink'
platter['#f3c07f'] = 'Orange'
platter['#0b1a22'] = 'Navy'
platter['#93afba'] = 'LightBlue'
platter['#128ecf'] = 'Blue'
platter['#717058'] = 'Green'
platter['#626e80'] = 'Navy'
platter['#7d7f8d'] = 'Gray'
platter['#c34e30'] = 'Orange'
platter['#128ecf'] = 'Blue'
platter['#218591'] = 'Aqua'
platter['#c70626'] = 'Red'
platter['#5f534d'] = 'Brown'
platter['#312d35'] = 'Black'
platter['#31559c'] = 'Blue'
platter['#e3b5a9'] = 'Pink'
platter['#eee4e6'] = 'Pink'
platter['#c1d8d3'] = 'Green'
platter['#d3d4c5'] = 'YellowGreen'
platter['#e8e7de'] = 'Ivory'
platter['#fb5028'] = 'Orange'
platter['#fda08e'] = 'Peach'
platter['#2a272e'] = 'Gray'
platter['#3e225b'] = 'Purple'
platter['#3d363f'] = 'Gray'
platter['#d2bfb5'] = 'SkinColor'
platter['#d8d2c7'] = 'Ivory'
platter['#2c2b2a'] = 'Black'
platter['#252428'] = 'Black'
platter['#222026'] = 'Black'
platter['#352e29'] = 'Black'
platter['#6c696d'] = 'Gray'
platter['#52bdc0'] = 'Aqua'
platter['#321d18'] = 'Brown'
platter['#332c26'] = 'Brown'
platter['#382c29'] = 'Brown'
platter['#f6f6c1'] = 'Yellow'
platter['#2b181c'] = 'Black'


lab_colors = []
rgb_color_keys = []
for rgb_color in platter.keys():
    hex = hex_to_rgb(rgb_color)
    rgb_color_keys.append(rgb_color)
    lab_colors.append(RGBColor(hex[0],hex[1],hex[2]).convert_to('lab'))

print lab_colors
print rgb_color_keys

def img_exist(url):
    try:
        req = urllib.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr,-1)
        img.shape
    except Exception, e:
        return 'no picture'
    else:
        return img

def shrink_img(img):
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
    print len(points)
    clusters = kmeans(points, n, 1)
    size = len(colors)/100
    sizes = []
    for x in range(n):
        sizes.append(len(clusters[x].points)/size)
    rgbs = [map(int, c.center.coords) for c in clusters]
    print 'percentage: ' + str(sizes)
    print 'Color(org): ' + str(map(rtoh, rgbs))
    # for x in range(n):
    #     save_color('test1', map(rtoh, rgbs)[x])
    return rgbs[sizes.index(max(sizes))]

def euclidean(p1, p2):
    return sqrt(sum([(p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 1
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    # print '----'
    # print vals
    # print plen
    # print points
    # print n
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

def get_bg_color(img, bg, skin):
    colors = []
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            if bg[r][c] == 255 and skin[r][c] == 0:
                colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    return colorz(colors, 3)

def get_colors(img, cbg, cnoskin):
    colors = []
    max_p = (img.shape[0] * img.shape[1])/5
    print max_p
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            if cnoskin[r][c] == cbg[r][c] == 0:
                colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
    print len(colors)
    if len(colors) > max_p:
        return colors
    else:
        return guess_colors(img, cbg, cnoskin)

def guess_colors(img, cbg, cnoskin):
    colors = []
    max_p = (img.shape[0] * img.shape[1])/2.5

    if len(colors) < max_p:
        colors = []
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cbg[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
        print 'no bg'
        print len(colors)

    if len(colors) < max_p:
        colors = []
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cnoskin[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
        print 'no skin'
        print len(colors)

    if len(colors) < max_p:
        colors = []
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
        print 'nothing'
        print len(colors)

    return colors



def get_rbg_color(p):
    distances = []
    p_lab = RGBColor(p[0],p[1],p[2])
    print len(lab_colors)
    for c in lab_colors:
        distances.append((p_lab).convert_to('lab').delta_e(c))
    print len(distances)
    print min(distances)
    print distances.index(min(distances))
    return hex_to_rgb(rgb_color_keys[distances.index(min(distances))])


platter_new = {}

def get_d_color(url):
    origin_img = img_exist(url)
    if origin_img == 'no picture':
        print 'No picture, man'
        return False
    else:
        try:
            img = shrink_img(origin_img)
            noskin = remove_skin(img)
            bg = remove_bg(img)
            # bg_color = get_bg_color(img, bg, noskin)
            cbg = crop(bg)
            cnoskin = crop(noskin)
            img = crop(img)
            # cv2.imwrite('cbg.png', cbg)
            # cv2.imwrite('cnoskin.png', cnoskin)
            # cv2.imwrite('img.png', img)
            colors = get_colors(img, cbg, cnoskin)
            img_colors = colorz(colors)
            rgb = get_rbg_color(img_colors)

            r = 300.0 / origin_img.shape[1]
            dim = (300, int(origin_img.shape[0] * r))
            origin_img = cv2.resize(origin_img, dim, interpolation = cv2.INTER_AREA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(origin_img,(15,15), 15, (img_colors[2],img_colors[1],img_colors[0]), -1)
            print rgb
            cv2.putText(origin_img,str(platter[rgb_to_hex(rgb)]),(30,25), font, 1,((rgb[2],rgb[1],rgb[0])),2)

            platter_new[rgb_to_hex((img_colors[0],img_colors[1],img_colors[2]))] = str(platter[rgb_to_hex(rgb)])
            print platter_new
            cv2.imwrite(rgb_to_hex((img_colors[0],img_colors[1],img_colors[2])) + '.jpg', origin_img)
        except:
            print 'something got wrong'



def get_styles(url):
    req = urllib.urlopen(url).read()
    result = json.loads(req)
    for x in range(30):
        # styles = result['styles'][x]['product']['styles']
        # for style in styles:
        #     print style['large_grid_image_url']
        #     get_d_color(style['large_grid_image_url'])
        #     print '==============================='
        #     print ''


        print result['styles'][x]['large_grid_image_url']
        get_d_color(result['styles'][x]['large_grid_image_url'])



def save_color(per, v):
    result = firebase.post('/colors/' + str(per), v)
    print result


if __name__ == "__main__":
    for page in range(26, 30):
        url = 'http://www.stylekick.com/api/v1/styles?color=red&gender=women&sort=trending&page='
        # print url + str(page)
        get_styles(url + str(page))


    # get_d_color('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/1645/large_grid_53_075680.jpg')



