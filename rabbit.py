import sys, json, time, traceback
import cv2
# from skimage import io
import urllib
import numpy as np
import pdb
from collections import namedtuple
from math import sqrt
import random
import json
from colormath.color_objects import RGBColor
from flask import Flask
from flask import request


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


platter = {'#fffff0': 'Ivory', '#009900': 'Green', '#F9F2DF': 'Beige', '#F5EBD0': 'Beige', '#3d363f': 'Gray', '#d8ee99': 'YellowGreen', '#FFFFD6': 'Ivory', '#C2E066': 'YellowGreen', '#666600': 'Yellow', '#009999': 'Aqua', '#8A00B8': 'Purple', '#004C4C': 'Aqua', '#2EB8E6': 'LightBlue', '#3e225b': 'Purple', '#B20000': 'Red', '#cfced9': 'Gray', '#c34e30': 'Orange', '#1F7A99': 'LightBlue', '#C266E0': 'Purple', '#FFFFDB': 'Ivory', '#A38566': 'Brown', '#E62E2E': 'Red', '#ffffff': 'White', '#2a272e': 'Gray', '#0000E6': 'Blue', '#5C007A': 'Purple', '#CC7A29': 'Orange', '#c88f6f': 'SkinColor', '#808000': 'Yellow', '#FF1919': 'Red', '#6a84a4': 'LightBlue', '#fda08e': 'Peach', '#000033': 'Blue', '#e8a1b1': 'Pink', '#c29582': 'SkinColor', '#0b1a22': 'Navy', '#CCCCA3': 'Ivory', '#CA8072': 'Peach', '#B2B28F': 'Ivory', '#717058': 'Green', '#FFFFD1': 'Ivory', '#85E0FF': 'LightBlue', '#E0B2F0': 'Purple', '#FFAD5C': 'Orange', '#f0eff1': 'White', '#99FFFF': 'Aqua', '#33FFFF': 'Aqua', '#202C3C': 'Navy', '#7E5047': 'Peach', '#4a4442': 'Black', '#7689c3': 'LightBlue', '#ffa500': 'Orange', '#F4E8C8': 'Beige', '#337074': 'Green', '#FFFFCC': 'Yellow', '#8080FF': 'Blue', '#33CCFF': 'LightBlue', '#FF87C3': 'Pink', '#D9DEE4': 'Navy', '#352e29': 'Black', '#FBF6F3': 'SkinColor', '#CCFFCC': 'Green', '#331F0A': 'Orange', '#B24A7E': 'Pink', '#B26B24': 'Orange', '#c8c8c3': 'Gray', '#6666FF': 'Blue', '#C1967B': 'SkinColor', '#33201C': 'Peach', '#19FFFF': 'Aqua', '#5C2E00': 'Brown', '#004C00': 'Green', '#00FFFF': 'Aqua', '#2f2e2a': 'Gray', '#c70626': 'Red', '#302C23': 'Beige', '#330A0A': 'Red', '#FED0C6': 'Peach', '#242b40': 'Navy', '#663300': 'Brown', '#804C1A': 'Orange', '#FFFF19': 'Yellow', '#312d35': 'Black', '#4DFF4D': 'Green', '#5f534d': 'Brown', '#6a96bf': 'LightBlue', '#FFA5D2': 'Pink', '#1F2900': 'YellowGreen', '#E6E600': 'Yellow', '#D1C2B2': 'Brown', '#FF8080': 'Red', '#f3efeb': 'Ivory', '#FFFF66': 'Yellow', '#993F6C': 'Pink', '#403229': 'SkinColor', '#B2FFB2': 'Green', '#F3E5DC': 'SkinColor', '#FFFF80': 'Yellow', '#472400': 'Brown', '#EFDCD0': 'SkinColor', '#330000': 'Red', '#EAD3C4': 'SkinColor', '#2E003D': 'Purple', '#273547': 'Navy', '#D699EB': 'Purple', '#2b181c': 'Red', '#e8e7de': 'Ivory', '#626e80': 'Navy', '#3D1F00': 'Brown', '#FDB3A5': 'Peach', '#FFF5EB': 'Orange', '#33FF33': 'Green', '#754719': 'Brown', '#123279': 'Blue', '#000099': 'Blue', '#F7EED8': 'Beige', '#fb5028': 'Orange', '#d4a694': 'SkinColor', '#A3D119': 'YellowGreen', '#E6E6FF': 'Blue', '#222026': 'Black', '#FAF5E7': 'Beige', '#27716d': 'Green', '#0F3D4C': 'LightBlue', '#f3c07f': 'Orange', '#00E600': 'Green', '#E2C1AC': 'SkinColor', '#52bdc0': 'Aqua', '#484335': 'Beige', '#70DBFF': 'LightBlue', '#EBFAFF': 'LightBlue', '#FFC3E1': 'Pink', '#0b3336': 'Green', '#edd7d0': 'SkinColor', '#FFB870': 'Orange', '#E49080': 'Peach', '#FFC285': 'Orange', '#B2B2FF': 'Blue', '#5CD6FF': 'LightBlue', '#4D4DFF': 'Blue', '#6250a1': 'Purple', '#DAB095': 'SkinColor', '#990000': 'Red', '#4C0066': 'Purple', '#99997A': 'Ivory', '#0D1218': 'Navy', '#ECEEF1': 'Navy', '#D6EB99': 'YellowGreen', '#6B5444': 'SkinColor', '#332c26': 'Brown', '#FFCC99': 'Orange', '#661414': 'Red', '#E6E6B8': 'Ivory', '#9b6649': 'SkinColor', '#806452': 'SkinColor', '#4C0F0F': 'Red', '#B2FFFF': 'Aqua', '#6b5b3f': 'Brown', '#FF4D4D': 'Red', '#FFE6E6': 'Red', '#80FFFF': 'Aqua', '#34465F': 'Navy', '#F0EBE6': 'Brown', '#80FF80': 'Green', '#522900': 'Brown', '#B2B200': 'Yellow', '#FDAA99': 'Peach', '#6e636f': 'Purple', '#FFD2E8': 'Pink', '#0000CC': 'Blue', '#4C4C00': 'Yellow', '#6B008F': 'Purple', '#B17063': 'Peach', '#A79B7B': 'Beige', '#128ecf': 'Blue', '#FFFFB2': 'Yellow', '#3333FF': 'Blue', '#2E3E53': 'Navy', '#231e1b': 'Black', '#003300': 'Green', '#4DFFFF': 'Aqua', '#29A3CC': 'LightBlue', '#B84DDB': 'Purple', '#331A00': 'Brown', '#FF4747': 'Red', '#E68A2E': 'Orange', '#FFFFF0': 'Ivory', '#99CC00': 'YellowGreen', '#654039': 'Peach', '#E60000': 'Red', '#B3BCC9': 'Navy', '#FFEBD6': 'Orange', '#66FF66': 'Green', '#31559c': 'Blue', '#d2bfb5': 'SkinColor', '#FFE0C2': 'Orange', '#47D1FF': 'LightBlue', '#969caa': 'Gray', '#E65EA2': 'Pink', '#000066': 'Blue', '#FF9999': 'Red', '#18171d': 'Black', '#93afba': 'LightBlue', '#000080': 'Blue', '#C2AD99': 'Brown', '#E6CAB8': 'SkinColor', '#A0ACBB': 'Navy', '#FDFCF7': 'Beige', '#967560': 'SkinColor', '#677992': 'Navy', '#A319D1': 'Purple', '#2c2b2a': 'Black', '#4C2E0F': 'Orange', '#546985': 'Navy', '#CC80E6': 'Purple', '#B8DB4D': 'YellowGreen', '#333300': 'Yellow', '#252428': 'Black', '#b4f8c1': 'YellowGreen', '#7A00A3': 'Purple', '#ADEBFF': 'LightBlue', '#f47f6e': 'Red', '#008000': 'Green', '#5B0124': 'Red', '#212b45': 'Navy', '#006666': 'Aqua', '#FFFFF5': 'Ivory', '#FFF0F8': 'Pink', '#efdeb0': 'Beige', '#ADD633': 'YellowGreen', '#AB866E': 'SkinColor', '#FF78BC': 'Pink', '#8AB800': 'YellowGreen', '#9289a8': 'Purple', '#dadbeb': 'Purple', '#181d29': 'Navy', '#1F0F00': 'Brown', '#FFFFFA': 'Ivory', '#d3d1d4': 'Gray', '#00B200': 'Green', '#1F0029': 'Purple', '#140A00': 'Brown', '#d3d4c5': 'YellowGreen', '#575a67': 'Gray', '#248FB2': 'LightBlue', '#FEBCB0': 'Peach', '#bfb3a3': 'Beige', '#382c29': 'Brown', '#3b352c': 'Green', '#574332': 'Brown', '#8193c0': 'LightBlue', '#FF96CA': 'Pink', '#3c3d4c': 'Navy', '#F5E6FA': 'Purple', '#FFB4DA': 'Pink', '#FFFF4D': 'Yellow', '#C2F0FF': 'LightBlue', '#F1E1B8': 'Beige', '#6B8F00': 'YellowGreen', '#d4cec9': 'Beige', '#FF5C5C': 'Red', '#e6ded2': 'Beige', '#331524': 'Pink', '#663D14': 'Orange', '#19FF19': 'Green', '#7A8AA0': 'Navy', '#6c696d': 'Gray', '#FFCCCC': 'Red', '#D6F5FF': 'LightBlue', '#ffff00': 'Yellow', '#FFFFEB': 'Ivory', '#028590': 'Green', '#986055': 'Peach', '#3D0052': 'Purple', '#4C4C3D': 'Ivory', '#898b2f': 'Green', '#4C6600': 'YellowGreen', '#00B2B2': 'Aqua', '#99E6FF': 'LightBlue', '#EBF5CC': 'YellowGreen', '#7e8c9b': 'Blue', '#FF9933': 'Orange', '#49474d': 'Gray', '#ff0000': 'Red', '#e3b5a9': 'Pink', '#415877': 'Navy', '#FFFFE0': 'Ivory', '#FFFFE6': 'Yellow', '#CCFFFF': 'Aqua', '#808066': 'Ivory', '#7d7f8d': 'Gray', '#EBCCF5': 'Purple', '#87ceeb': 'LightBlue', '#d1ab99': 'SkinColor', '#800000': 'Red', '#786F58': 'Beige', '#995C1F': 'Orange', '#291400': 'Brown', '#808080': 'Gray', '#FFFF33': 'Yellow', '#d6a789': 'SkinColor', '#CCCCFF': 'Blue', '#2E3D00': 'YellowGreen', '#E0F0B2': 'YellowGreen', '#1f61c3': 'Blue', '#0000B2': 'Blue', '#FF0000': 'Red', '#00CCCC': 'Aqua', '#1A2330': 'Navy', '#FED9D2': 'Peach', '#FEE2DD': 'Peach', '#7c8daa': 'Navy', '#2B211B': 'SkinColor', '#855C33': 'Brown', '#3D5200': 'YellowGreen', '#000000': 'Black', '#FF69B4': 'Pink', '#FCF8EF': 'Beige', '#BFB28D': 'Beige', '#218591': 'Aqua', '#145266': 'LightBlue', '#FEC6BB': 'Peach', '#001A1A': 'Aqua', '#660000': 'Red', '#DEB9A1': 'SkinColor', '#B29980': 'Brown', '#CCCC00': 'Yellow', '#f6f6c1': 'Yellow', '#CC5490': 'Pink', '#4C302B': 'Peach', '#FFE1F0': 'Pink', '#dec6c1': 'Pink', '#9999FF': 'Blue', '#801A1A': 'Red', '#008080': 'Aqua', '#FFF6F4': 'Peach', '#99FF99': 'Green', '#8F856A': 'Beige', '#94704D': 'Brown', '#D7C89E': 'Beige', '#E6FFFF': 'Aqua', '#321d18': 'Brown', '#683300': 'Brown', '#141A24': 'Navy', '#00FF00': 'Green', '#FF3333': 'Red', '#564337': 'SkinColor', '#8D9BAD': 'Navy', '#666652': 'Ivory', '#eee4e6': 'Pink', '#FFD6AD': 'Orange', '#e8a79d': 'Pink', '#FFECE8': 'Peach', '#CC0000': 'Red', '#F7EDE7': 'SkinColor', '#999900': 'Yellow', '#80345A': 'Pink', '#991F1F': 'Red', '#662A48': 'Pink', '#e7e7f0': 'White', '#0000ff': 'Blue', '#C6CDD6': 'Navy', '#003333': 'Aqua', '#00ffff': 'Aqua', '#006600': 'Green', '#00E6E6': 'Aqua', '#F2E5C0': 'Beige', '#1f2a33': 'Black', '#cfdee8': 'LightBlue', '#1919FF': 'Blue', '#4C0000': 'Red', '#605946': 'Beige', '#0000FF': 'Blue', '#FF6666': 'Red', '#d4efb6': 'YellowGreen', '#5b5957': 'Gray', '#9900CC': 'Purple', '#CCE680': 'YellowGreen', '#d8d2c7': 'Ivory', '#AD33D6': 'Purple', '#FFA347': 'Orange', '#8800CC': 'Purple', '#FFFF99': 'Yellow', '#5C7A00': 'YellowGreen', '#E0D6CC': 'Brown', '#c1d8d3': 'Green', '#B22424': 'Red', '#1A6680': 'LightBlue', '#0A2933': 'LightBlue', '#66FFFF': 'Aqua', '#00004C': 'Blue', '#4C2036': 'Pink', '#FFB2B2': 'Red', '#CC2929': 'Red', '#9acd32': 'YellowGreen', '#947f75': 'Beige', '#00CC00': 'Green', '#7AA300': 'YellowGreen', '#3A4F6B': 'Navy'}


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
    print 'i am here'

    new_img = np.zeros((img.shape[0],img.shape[1],4), np.uint8)
    cv2.imwrite('remove_skin_mask.jpg', mask)
    cv2.imwrite('remove_skin_zeros.jpg', new_img)
    cv2.imwrite('remove_skin_skin_ycrcb.jpg', skin_ycrcb)


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
            cv2.imwrite('cbg.png', cbg)
            cv2.imwrite('cnoskin.png', cnoskin)
            cv2.imwrite('img.png', img)
            print ' === === process ==='
            colors = get_colors(img, cbg, cnoskin)
            print 'color: ' + str(color)
            img_colors = colorz(colors)
            print str(img_colors)
            rgb = get_rbg_color(img_colors)
            print 'hex: ' + str(platter[rgb_to_hex(rgb)])
            return platter[rgb_to_hex(rgb)]

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


app = Flask(__name__)



if __name__ == "__main__":
    # for page in range(26, 30):
    #     url = 'http://www.stylekick.com/api/v1/styles?color=red&gender=women&sort=trending&page='
    #     # print url + str(page)
    #     get_styles(url + str(page))


    get_d_color('https://stylekick-assets.s3.amazonaws.com/uploads/style/image/1645/large_grid_53_075680.jpg')
    # @app.route('/detect', methods=['POST'])
    # def detect():
    #     url = json.loads(request.data)['url']
    #     color = get_d_color(url)
    #     print color
    #     return color

    # # app.debug = True
    # app.run()
