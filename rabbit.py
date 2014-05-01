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
from flask import Flask
from flask import request

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
platter['#7e8c9b'] = 'Blue'
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
platter['#123279'] = 'Blue'
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
platter['#2b181c'] = 'Red'


platter['#331F0A'] = 'Orange'
platter['#4C2E0F'] = 'Orange'
platter['#663D14'] = 'Orange'
platter['#804C1A'] = 'Orange'
platter['#995C1F'] = 'Orange'
platter['#B26B24'] = 'Orange'
platter['#CC7A29'] = 'Orange'
platter['#E68A2E'] = 'Orange'
platter['#FF9933'] = 'Orange'
platter['#FFA347'] = 'Orange'
platter['#FFAD5C'] = 'Orange'
platter['#FFB870'] = 'Orange'
platter['#FFC285'] = 'Orange'
platter['#FFCC99'] = 'Orange'
platter['#FFD6AD'] = 'Orange'
platter['#FFE0C2'] = 'Orange'
platter['#FFEBD6'] = 'Orange'
platter['#FFF5EB'] = 'Orange'

platter['#330A0A'] = 'Red'
platter['#4C0F0F'] = 'Red'
platter['#661414'] = 'Red'
platter['#801A1A'] = 'Red'
platter['#991F1F'] = 'Red'
platter['#B22424'] = 'Red'
platter['#CC2929'] = 'Red'
platter['#E62E2E'] = 'Red'
platter['#FF3333'] = 'Red'
platter['#FF4747'] = 'Red'
platter['#FF5C5C'] = 'Red'

platter['#003300'] = 'Green'
platter['#004C00'] = 'Green'
platter['#006600'] = 'Green'
platter['#008000'] = 'Green'
platter['#009900'] = 'Green'
platter['#00B200'] = 'Green'
platter['#00CC00'] = 'Green'
platter['#00E600'] = 'Green'
platter['#00FF00'] = 'Green'
platter['#19FF19'] = 'Green'
platter['#33FF33'] = 'Green'
platter['#4DFF4D'] = 'Green'
platter['#66FF66'] = 'Green'
platter['#80FF80'] = 'Green'
platter['#99FF99'] = 'Green'
platter['#B2FFB2'] = 'Green'
platter['#CCFFCC'] = 'Green'

platter['#1F2900'] = 'YellowGreen'
platter['#2E3D00'] = 'YellowGreen'
platter['#3D5200'] = 'YellowGreen'
platter['#4C6600'] = 'YellowGreen'
platter['#5C7A00'] = 'YellowGreen'
platter['#6B8F00'] = 'YellowGreen'
platter['#7AA300'] = 'YellowGreen'
platter['#8AB800'] = 'YellowGreen'
platter['#99CC00'] = 'YellowGreen'
platter['#A3D119'] = 'YellowGreen'
platter['#ADD633'] = 'YellowGreen'
platter['#B8DB4D'] = 'YellowGreen'
platter['#C2E066'] = 'YellowGreen'
platter['#CCE680'] = 'YellowGreen'
platter['#D6EB99'] = 'YellowGreen'
platter['#E0F0B2'] = 'YellowGreen'
platter['#EBF5CC'] = 'YellowGreen'

platter['#4C4C3D'] = 'Ivory'
platter['#666652'] = 'Ivory'
platter['#808066'] = 'Ivory'
platter['#99997A'] = 'Ivory'
platter['#B2B28F'] = 'Ivory'
platter['#CCCCA3'] = 'Ivory'
platter['#E6E6B8'] = 'Ivory'
platter['#FFFFCC'] = 'Ivory'
platter['#FFFFD1'] = 'Ivory'
platter['#FFFFD6'] = 'Ivory'
platter['#FFFFDB'] = 'Ivory'
platter['#FFFFE0'] = 'Ivory'
platter['#FFFFE6'] = 'Ivory'
platter['#FFFFEB'] = 'Ivory'
platter['#FFFFF0'] = 'Ivory'
platter['#FFFFF5'] = 'Ivory'
platter['#FFFFFA'] = 'Ivory'

platter['#001A1A'] = 'Aqua'
platter['#003333'] = 'Aqua'
platter['#004C4C'] = 'Aqua'
platter['#006666'] = 'Aqua'
platter['#008080'] = 'Aqua'
platter['#009999'] = 'Aqua'
platter['#00B2B2'] = 'Aqua'
platter['#00CCCC'] = 'Aqua'
platter['#00E6E6'] = 'Aqua'
platter['#00FFFF'] = 'Aqua'
platter['#19FFFF'] = 'Aqua'
platter['#33FFFF'] = 'Aqua'
platter['#4DFFFF'] = 'Aqua'
platter['#66FFFF'] = 'Aqua'
platter['#80FFFF'] = 'Aqua'
platter['#99FFFF'] = 'Aqua'
platter['#B2FFFF'] = 'Aqua'
platter['#CCFFFF'] = 'Aqua'
platter['#E6FFFF'] = 'Aqua'

platter['#0A2933'] = 'LightBlue'
platter['#0F3D4C'] = 'LightBlue'
platter['#145266'] = 'LightBlue'
platter['#1A6680'] = 'LightBlue'
platter['#1F7A99'] = 'LightBlue'
platter['#248FB2'] = 'LightBlue'
platter['#29A3CC'] = 'LightBlue'
platter['#2EB8E6'] = 'LightBlue'
platter['#33CCFF'] = 'LightBlue'
platter['#47D1FF'] = 'LightBlue'
platter['#5CD6FF'] = 'LightBlue'
platter['#70DBFF'] = 'LightBlue'
platter['#85E0FF'] = 'LightBlue'
platter['#99E6FF'] = 'LightBlue'
platter['#ADEBFF'] = 'LightBlue'
platter['#C2F0FF'] = 'LightBlue'
platter['#D6F5FF'] = 'LightBlue'
platter['#EBFAFF'] = 'LightBlue'


platter['#1F0029'] = 'Purple'
platter['#2E003D'] = 'Purple'
platter['#3D0052'] = 'Purple'
platter['#4C0066'] = 'Purple'
platter['#5C007A'] = 'Purple'
platter['#6B008F'] = 'Purple'
platter['#7A00A3'] = 'Purple'
platter['#8A00B8'] = 'Purple'
platter['#9900CC'] = 'Purple'
platter['#A319D1'] = 'Purple'
platter['#AD33D6'] = 'Purple'
platter['#B84DDB'] = 'Purple'
platter['#C266E0'] = 'Purple'
platter['#CC80E6'] = 'Purple'
platter['#D699EB'] = 'Purple'
platter['#E0B2F0'] = 'Purple'
platter['#EBCCF5'] = 'Purple'
platter['#F5E6FA'] = 'Purple'


platter['#000033'] = 'Blue'
platter['#00004C'] = 'Blue'
platter['#000066'] = 'Blue'
platter['#000080'] = 'Blue'
platter['#000099'] = 'Blue'
platter['#0000B2'] = 'Blue'
platter['#0000CC'] = 'Blue'
platter['#0000E6'] = 'Blue'
platter['#0000FF'] = 'Blue'
platter['#1919FF'] = 'Blue'
platter['#3333FF'] = 'Blue'
platter['#4D4DFF'] = 'Blue'
platter['#6666FF'] = 'Blue'
platter['#8080FF'] = 'Blue'
platter['#9999FF'] = 'Blue'
platter['#B2B2FF'] = 'Blue'
platter['#CCCCFF'] = 'Blue'
platter['#E6E6FF'] = 'Blue'

platter['#140A00'] = 'Brown'
platter['#1F0F00'] = 'Brown'
platter['#291400'] = 'Brown'
platter['#331A00'] = 'Brown'
platter['#3D1F00'] = 'Brown'
platter['#472400'] = 'Brown'
platter['#522900'] = 'Brown'
platter['#5C2E00'] = 'Brown'
platter['#663300'] = 'Brown'
platter['#754719'] = 'Brown'
platter['#855C33'] = 'Brown'
platter['#94704D'] = 'Brown'
platter['#A38566'] = 'Brown'
platter['#B29980'] = 'Brown'
platter['#C2AD99'] = 'Brown'
platter['#D1C2B2'] = 'Brown'
platter['#E0D6CC'] = 'Brown'
platter['#F0EBE6'] = 'Brown'


platter['#302C23'] = 'Beige'
platter['#484335'] = 'Beige'
platter['#605946'] = 'Beige'
platter['#786F58'] = 'Beige'
platter['#8F856A'] = 'Beige'
platter['#A79B7B'] = 'Beige'
platter['#BFB28D'] = 'Beige'
platter['#D7C89E'] = 'Beige'
platter['#efdeb0'] = 'Beige'
platter['#F1E1B8'] = 'Beige'
platter['#F2E5C0'] = 'Beige'
platter['#F4E8C8'] = 'Beige'
platter['#F5EBD0'] = 'Beige'
platter['#F7EED8'] = 'Beige'
platter['#F9F2DF'] = 'Beige'
platter['#FAF5E7'] = 'Beige'
platter['#FCF8EF'] = 'Beige'
platter['#FDFCF7'] = 'Beige'

platter['#330000'] = 'Red'
platter['#4C0000'] = 'Red'
platter['#660000'] = 'Red'
platter['#800000'] = 'Red'
platter['#990000'] = 'Red'
platter['#B20000'] = 'Red'
platter['#CC0000'] = 'Red'
platter['#E60000'] = 'Red'
platter['#FF0000'] = 'Red'
platter['#FF1919'] = 'Red'
platter['#FF3333'] = 'Red'
platter['#FF4D4D'] = 'Red'
platter['#FF6666'] = 'Red'
platter['#FF8080'] = 'Red'
platter['#FF9999'] = 'Red'
platter['#FFB2B2'] = 'Red'
platter['#FFCCCC'] = 'Red'
platter['#FFE6E6'] = 'Red'

platter['#333300'] = 'Yellow'
platter['#4C4C00'] = 'Yellow'
platter['#666600'] = 'Yellow'
platter['#808000'] = 'Yellow'
platter['#999900'] = 'Yellow'
platter['#B2B200'] = 'Yellow'
platter['#CCCC00'] = 'Yellow'
platter['#E6E600'] = 'Yellow'
platter['#ffff00'] = 'Yellow'
platter['#FFFF19'] = 'Yellow'
platter['#FFFF33'] = 'Yellow'
platter['#FFFF4D'] = 'Yellow'
platter['#FFFF66'] = 'Yellow'
platter['#FFFF80'] = 'Yellow'
platter['#FFFF99'] = 'Yellow'
platter['#FFFFB2'] = 'Yellow'
platter['#FFFFCC'] = 'Yellow'
platter['#FFFFE6'] = 'Yellow'

platter['#331524'] = 'Pink'
platter['#4C2036'] = 'Pink'
platter['#662A48'] = 'Pink'
platter['#80345A'] = 'Pink'
platter['#993F6C'] = 'Pink'
platter['#B24A7E'] = 'Pink'
platter['#CC5490'] = 'Pink'
platter['#E65EA2'] = 'Pink'
platter['#FF69B4'] = 'Pink'
platter['#FF78BC'] = 'Pink'
platter['#FF87C3'] = 'Pink'
platter['#FF96CA'] = 'Pink'
platter['#FFA5D2'] = 'Pink'
platter['#FFB4DA'] = 'Pink'
platter['#FFC3E1'] = 'Pink'
platter['#FFD2E8'] = 'Pink'
platter['#FFE1F0'] = 'Pink'
platter['#FFF0F8'] = 'Pink'

platter['#2B211B'] = 'SkinColor'
platter['#403229'] = 'SkinColor'
platter['#564337'] = 'SkinColor'
platter['#6B5444'] = 'SkinColor'
platter['#806452'] = 'SkinColor'
platter['#967560'] = 'SkinColor'
platter['#AB866E'] = 'SkinColor'
platter['#C1967B'] = 'SkinColor'
platter['#d6a789'] = 'SkinColor'
platter['#DAB095'] = 'SkinColor'
platter['#DEB9A1'] = 'SkinColor'
platter['#E2C1AC'] = 'SkinColor'
platter['#E6CAB8'] = 'SkinColor'
platter['#EAD3C4'] = 'SkinColor'
platter['#EFDCD0'] = 'SkinColor'
platter['#F3E5DC'] = 'SkinColor'
platter['#F7EDE7'] = 'SkinColor'
platter['#FBF6F3'] = 'SkinColor'

platter['#0D1218'] = 'Navy'
platter['#141A24'] = 'Navy'
platter['#1A2330'] = 'Navy'
platter['#202C3C'] = 'Navy'
platter['#273547'] = 'Navy'
platter['#2E3E53'] = 'Navy'
platter['#34465F'] = 'Navy'
platter['#3A4F6B'] = 'Navy'
platter['#415877'] = 'Navy'
platter['#546985'] = 'Navy'
platter['#677992'] = 'Navy'
platter['#7A8AA0'] = 'Navy'
platter['#8D9BAD'] = 'Navy'
platter['#A0ACBB'] = 'Navy'
platter['#B3BCC9'] = 'Navy'
platter['#C6CDD6'] = 'Navy'
platter['#D9DEE4'] = 'Navy'
platter['#ECEEF1'] = 'Navy'

platter['#33201C'] = 'Peach'
platter['#4C302B'] = 'Peach'
platter['#654039'] = 'Peach'
platter['#7E5047'] = 'Peach'
platter['#986055'] = 'Peach'
platter['#B17063'] = 'Peach'
platter['#CA8072'] = 'Peach'
platter['#E49080'] = 'Peach'
platter['#fda08e'] = 'Peach'
platter['#FDAA99'] = 'Peach'
platter['#FDB3A5'] = 'Peach'
platter['#FEBCB0'] = 'Peach'
platter['#FEC6BB'] = 'Peach'
platter['#FED0C6'] = 'Peach'
platter['#FED9D2'] = 'Peach'
platter['#FEE2DD'] = 'Peach'
platter['#FFECE8'] = 'Peach'
platter['#FFF6F4'] = 'Peach'

lab_colors = []
rgb_color_keys = []
for rgb_color in platter.keys():
    hex = hex_to_rgb(rgb_color)
    rgb_color_keys.append(rgb_color)
    lab_colors.append(RGBColor(hex[0],hex[1],hex[2]).convert_to('lab'))

# print lab_colors
# print rgb_color_keys

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
            print ' === === process ==='
            colors = get_colors(img, cbg, cnoskin)
            print 'color: ' + str(color)
            img_colors = colorz(colors)
            print str(img_colors)
            rgb = get_rbg_color(img_colors)
            print 'hex: ' + str(platter[rgb_to_hex(rgb)])
            return platter[rgb_to_hex(rgb)]
            # r = 300.0 / origin_img.shape[1]
            # dim = (300, int(origin_img.shape[0] * r))
            # origin_img = cv2.resize(origin_img, dim, interpolation = cv2.INTER_AREA)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.circle(origin_img,(15,15), 15, (img_colors[2],img_colors[1],img_colors[0]), -1)
            # print rgb
            # cv2.putText(origin_img,str(platter[rgb_to_hex(rgb)]),(30,25), font, 1,((rgb[2],rgb[1],rgb[0])),2)

            # platter_new[rgb_to_hex((img_colors[0],img_colors[1],img_colors[2]))] = str(platter[rgb_to_hex(rgb)])
            # print platter_new
            # cv2.imwrite(rgb_to_hex((img_colors[0],img_colors[1],img_colors[2])) + '.jpg', origin_img)
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


app = Flask(__name__)



if __name__ == "__main__":
    # for page in range(26, 30):
    #     url = 'http://www.stylekick.com/api/v1/styles?color=red&gender=women&sort=trending&page='
    #     # print url + str(page)
    #     get_styles(url + str(page))


    # get_d_color('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/1645/large_grid_53_075680.jpg')
    @app.route('/detect', methods=['POST'])
    def detect():
        url = json.loads(request.data)['url']
        color = get_d_color(url)
        print color
        return color

    # app.debug = True
    app.run()
