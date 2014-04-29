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

firebase = firebase.FirebaseApplication('https://stylekick-colors.firebaseio.com/', None)

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
    #     save_color(sizes[x], map(rtoh, rgbs)[x])
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
                if cnoskin[r][c] == 0:
                    colors.append((int(img[r][c][2]),int(img[r][c][1]),int(img[r][c][0])))
        print 'no bg'
        print len(colors)

    if len(colors) < max_p:
        colors = []
        for r in range(0,img.shape[0]):
            for c in range(0,img.shape[1]):
                if cbg[r][c] == 0:
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

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def get_color_name(p):
    platter = {}
    colors['#f0f8ff'] = 'AliceBlue'
    colors['#faebd7'] = 'AntiqueWhite'
    colors['#00ffff'] = 'Aqua'
    colors['#7fffd4'] = 'Aquamarine'
    colors['#f0ffff'] = 'Azure'
    colors['#f5f5dc'] = 'Beige'
    colors['#ffe4c4'] = 'Bisque'
    colors['#000000'] = 'Black'
    colors['#ffebcd'] = 'BlanchedAlmond'
    colors['#0000ff'] = 'Blue'
    colors['#8a2be2'] = 'BlueViolet'
    colors['#a52a2a'] = 'Brown'
    colors['#deb887'] = 'BurlyWood'
    colors['#5f9ea0'] = 'CadetBlue'
    colors['#7fff00'] = 'Chartreuse'
    colors['#d2691e'] = 'Chocolate'
    colors['#ff7f50'] = 'Coral'
    colors['#6495ed'] = 'CornflowerBlue'
    colors['#fff8dc'] = 'Cornsilk'
    colors['#dc143c'] = 'Crimson'
    colors['#00ffff'] = 'Cyan'
    colors['#00008b'] = 'DarkBlue'
    colors['#008b8b'] = 'DarkCyan'
    colors['#b8860b'] = 'DarkGoldenRod'
    colors['#a9a9a9'] = 'DarkGray'
    colors['#006400'] = 'DarkGreen'
    colors['#bdb76b'] = 'DarkKhaki'
    colors['#8b008b'] = 'DarkMagenta'
    colors['#556b2f'] = 'DarkOliveGreen'
    colors['#ff8c00'] = 'DarkOrange'
    colors['#9932cc'] = 'DarkOrchid'
    colors['#8b0000'] = 'DarkRed'
    colors['#e9967a'] = 'DarkSalmon'
    colors['#8fbc8f'] = 'DarkSeaGreen'
    colors['#483d8b'] = 'DarkSlateBlue'
    colors['#2f4f4f'] = 'DarkSlateGray'
    colors['#00ced1'] = 'DarkTurquoise'
    colors['#9400d3'] = 'DarkViolet'
    colors['#ff1493'] = 'DeepPink'
    colors['#00bfff'] = 'DeepSkyBlue'
    colors['#696969'] = 'DimGray'
    colors['#1e90ff'] = 'DodgerBlue'
    colors['#b22222'] = 'FireBrick'
    colors['#fffaf0'] = 'FloralWhite'
    colors['#228b22'] = 'ForestGreen'
    colors['#ff00ff'] = 'Fuchsia'
    colors['#dcdcdc'] = 'Gainsboro'
    colors['#f8f8ff'] = 'GhostWhite'
    colors['#ffd700'] = 'Gold'
    colors['#daa520'] = 'GoldenRod'
    colors['#808080'] = 'Gray'
    colors['#008000'] = 'Green'
    colors['#adff2f'] = 'GreenYellow'
    colors['#f0fff0'] = 'HoneyDew'
    colors['#ff69b4'] = 'HotPink'
    colors['#cd5c5c'] = 'IndianRed'
    colors['#4b0082'] = 'Indigo'
    colors['#fffff0'] = 'Ivory'
    colors['#f0e68c'] = 'Khaki'
    colors['#e6e6fa'] = 'Lavender'
    colors['#fff0f5'] = 'LavenderBlush'
    colors['#7cfc00'] = 'LawnGreen'
    colors['#fffacd'] = 'LemonChiffon'
    colors['#add8e6'] = 'LightBlue'
    colors['#f08080'] = 'LightCoral'
    colors['#e0ffff'] = 'LightCyan'
    colors['#fafad2'] = 'LightGoldenRodYellow'
    colors['#d3d3d3'] = 'LightGray'
    colors['#90ee90'] = 'LightGreen'
    colors['#ffb6c1'] = 'LightPink'
    colors['#ffa07a'] = 'LightSalmon'
    colors['#20b2aa'] = 'LightSeaGreen'
    colors['#87cefa'] = 'LightSkyBlue'
    colors['#778899'] = 'LightSlateGray'
    colors['#b0c4de'] = 'LightSteelBlue'
    colors['#ffffe0'] = 'LightYellow'
    colors['#00ff00'] = 'Lime'
    colors['#32cd32'] = 'LimeGreen'
    colors['#faf0e6'] = 'Linen'
    colors['#ff00ff'] = 'Magenta'
    colors['#800000'] = 'Maroon'
    colors['#66cdaa'] = 'MediumAquaMarine'
    colors['#0000cd'] = 'MediumBlue'
    colors['#ba55d3'] = 'MediumOrchid'
    colors['#9370db'] = 'MediumPurple'
    colors['#3cb371'] = 'MediumSeaGreen'
    colors['#7b68ee'] = 'MediumSlateBlue'
    colors['#00fa9a'] = 'MediumSpringGreen'
    colors['#48d1cc'] = 'MediumTurquoise'
    colors['#c71585'] = 'MediumVioletRed'
    colors['#191970'] = 'MidnightBlue'
    colors['#f5fffa'] = 'MintCream'
    colors['#ffe4e1'] = 'MistyRose'
    colors['#ffe4b5'] = 'Moccasin'
    colors['#ffdead'] = 'NavajoWhite'
    colors['#000080'] = 'Navy'
    colors['#fdf5e6'] = 'OldLace'
    colors['#808000'] = 'Olive'
    colors['#6b8e23'] = 'OliveDrab'
    colors['#ffa500'] = 'Orange'
    colors['#ff4500'] = 'OrangeRed'
    colors['#da70d6'] = 'Orchid'
    colors['#eee8aa'] = 'PaleGoldenRod'
    colors['#98fb98'] = 'PaleGreen'
    colors['#afeeee'] = 'PaleTurquoise'
    colors['#db7093'] = 'PaleVioletRed'
    colors['#ffefd5'] = 'PapayaWhip'
    colors['#ffdab9'] = 'PeachPuff'
    colors['#cd853f'] = 'Peru'
    colors['#ffc0cb'] = 'Pink'
    colors['#dda0dd'] = 'Plum'
    colors['#b0e0e6'] = 'PowderBlue'
    colors['#800080'] = 'Purple'
    colors['#ff0000'] = 'Red'
    colors['#bc8f8f'] = 'RosyBrown'
    colors['#4169e1'] = 'RoyalBlue'
    colors['#8b4513'] = 'SaddleBrown'
    colors['#fa8072'] = 'Salmon'
    colors['#f4a460'] = 'SandyBrown'
    colors['#2e8b57'] = 'SeaGreen'
    colors['#fff5ee'] = 'SeaShell'
    colors['#a0522d'] = 'Sienna'
    colors['#c0c0c0'] = 'Silver'
    colors['#87ceeb'] = 'SkyBlue'
    colors['#6a5acd'] = 'SlateBlue'
    colors['#708090'] = 'SlateGray'
    colors['#fffafa'] = 'Snow'
    colors['#00ff7f'] = 'SpringGreen'
    colors['#4682b4'] = 'SteelBlue'
    colors['#d2b48c'] = 'Tan'
    colors['#008080'] = 'Teal'
    colors['#d8bfd8'] = 'Thistle'
    colors['#ff6347'] = 'Tomato'
    colors['#40e0d0'] = 'Turquoise'
    colors['#ee82ee'] = 'Violet'
    colors['#f5deb3'] = 'Wheat'
    colors['#ffffff'] = 'White'
    colors['#f5f5f5'] = 'WhiteSmoke'
    colors['#ffff00'] = 'Yellow'
    colors['#9acd32'] = 'YellowGreen'
    rgbs = [(128, 128, 128), (173, 255, 47), (255, 245, 238), (85, 107, 47), (255, 140, 0), (153, 50, 204), (138, 43, 226), (186, 85, 211), (47, 79, 79), (0, 0, 139), (219, 112, 147), (0, 0, 255), (220, 20, 60), (221, 160, 221), (65, 105, 225), (218, 112, 214), (220, 220, 220), (95, 158, 160), (147, 112, 219), (175, 238, 238), (230, 230, 250), (0, 0, 0), (107, 142, 35), (255, 105, 180), (240, 248, 255), (34, 139, 34), (0, 139, 139), (255, 127, 80), (238, 130, 238), (211, 211, 211), (255, 0, 255), (72, 209, 204), (255, 255, 255), (245, 222, 179), (0, 250, 154), (240, 128, 128), (128, 128, 0), (250, 235, 215), (169, 169, 169), (127, 255, 212), (192, 192, 192), (127, 255, 0), (255, 235, 205), (176, 196, 222), (119, 136, 153), (255, 250, 205), (255, 215, 0), (0, 128, 0), (139, 69, 19), (255, 240, 245), (255, 255, 240), (106, 90, 205), (128, 0, 128), (255, 250, 250), (70, 130, 180), (255, 239, 213), (238, 232, 170), (0, 255, 0), (255, 222, 173), (205, 133, 63), (173, 216, 230), (152, 251, 152), (224, 255, 255), (248, 248, 255), (216, 191, 216), (188, 143, 143), (255, 0, 0), (0, 0, 128), (0, 206, 209), (160, 82, 45), (255, 192, 203), (123, 104, 238), (205, 92, 92), (46, 139, 87), (184, 134, 11), (255, 160, 122), (64, 224, 208), (250, 250, 210), (222, 184, 135), (240, 255, 255), (255, 248, 220), (233, 150, 122), (135, 206, 235), (25, 25, 112), (144, 238, 144), (210, 180, 140), (0, 255, 255), (0, 0, 205), (124, 252, 0), (255, 228, 225), (189, 183, 107), (244, 164, 96), (240, 255, 240), (60, 179, 113), (245, 255, 250), (32, 178, 170), (30, 144, 255), (112, 128, 144), (245, 245, 220), (102, 205, 170), (154, 205, 50), (199, 21, 133), (245, 245, 245), (50, 205, 50), (139, 0, 0), (105, 105, 105), (148, 0, 211), (143, 188, 143), (0, 191, 255), (72, 61, 139), (100, 149, 237), (255, 165, 0), (0, 255, 127), (165, 42, 42), (250, 240, 230), (0, 128, 128), (255, 228, 181), (255, 99, 71), (178, 34, 34), (218, 165, 32), (75, 0, 130), (255, 250, 240), (176, 224, 230), (240, 230, 140), (255, 255, 0), (0, 100, 0), (255, 228, 196), (253, 245, 230), (139, 0, 139), (255, 255, 224), (250, 128, 114), (255, 218, 185), (210, 105, 30), (255, 20, 147), (255, 182, 193), (128, 0, 0), (135, 206, 250), (255, 69, 0)]
    distances = []
    for c in rgbs:
        distances.append(int(sqrt(sum([(c[i] - p[i]) ** 2 for i in range(3)]))))
    # print distances
    # print min(distances)
    # print distances.index(min(distances))
    return platter[rgb_to_hex(rgbs[distances.index(min(distances))])]

def compare_color(bg, img):
    pass

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
            name = get_color_name(img_colors)

            r = 300.0 / origin_img.shape[1]
            dim = (300, int(origin_img.shape[0] * r))
            origin_img = cv2.resize(origin_img, dim, interpolation = cv2.INTER_AREA)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(origin_img,str(name),(10,25), font, 1,(img_colors[2],img_colors[1],img_colors[0]),2)

            cv2.imwrite(url.split('/')[-1] + '.png', origin_img)
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
    for page in range(4, 100):
        url = 'http://www.stylekick.com/api/v1/styles?gender=women&sort=trending&page='
        # print url + str(page)
        get_styles(url + str(page))


    # get_d_color('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/108546/large_grid_7d43975e6f83a2edd6d845804058f822_best.jpg')





