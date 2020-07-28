#/usr/bin/python3
#-*- encoding=utf-8 -*-
from alchemy_config import cfg
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.quad.get_quad_roi import get_quad_roi
from utils.resize_img import *
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s')

def drawRect(img, quads, color, lineWidth):
    pt1, pt2, pt3, pt4 = [(int(quads[i]), int(quads[i+1])) for i in [0, 2, 4, 6]]
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)
    return img

def test4get_quad_roi():
    img_path = os.path.join(cfg.ROOT_DIR, 'tests/testimg/TB1izNTLXXXXXa.XXXXunYpLFXX.jpg')
    logging.info("Test for uitls --> get quad roi")
    quads = [299.30374, 624.38715, 449, 530, 458, 546, 310.56863, 645.8275]
    img = cv2.imread(img_path)
    if img is None:
        logging.error("img is None, check img path: %s" %(img_path))
        return
    roi = get_quad_roi(quads, img)
    quad_roi_ori = drawRect(img, quads, (0, 0, 255), 2)

    axes1 = plt.subplot(121)
    axes1.set_title('quad_roi_ori', fontstyle='italic')
    plt.imshow(quad_roi_ori)

    axes2 = plt.subplot(122)
    axes2.set_title('quad_roi_roted', fontstyle='italic')
    plt.imshow(roi)
    plt.show()

def test4resize_img():
    def check_resize_self(resizefunc, img):
        h, w = img.shape[: 2]
        resize_self = resizefunc(img, h, w)
        difference = cv2.subtract(img, resize_self)
        result = not np.any(difference)
        if result:
            logging.info("-> %s resize itself is ok!" %(resizefunc.__name__))
        else:
            logging.info("-> %s resize itself is error!" %(resizefunc.__name__))
            plt.imshow(difference)
            plt.show()
    def check_half_h_resize(resizefunc, img):
        h, w = img.shape[: 2]
        resized_img = resizefunc(img, h/2, w)
        plt.ion()
        axes1 = plt.subplot(121)
        axes1.set_title('ori', fontstyle='italic')
        plt.imshow(img)
        axes2 = plt.subplot(122)
        axes2.set_title('resized_img', fontstyle='italic')
        plt.imshow(resized_img)
        plt.pause(10)
        plt.close()
    
    img_path = os.path.join(cfg.ROOT_DIR, 'tests/testimg/0A4DSPGE.jpg')
    img = cv2.imread(img_path)
    for item in [undeform_resize, undeform_center_resize, resize_with_center_pad]:
        # check_resize_self(item, img)
        check_half_h_resize(item, img)



def main():
    # logging.info("========== Test for utils-quad ==========")
    # test4get_quad_roi()
    logging.info("========== Test for resize_img ==========")
    test4resize_img()

if __name__ == "__main__":
    logging.info("========== Test for utils! ==========")
    main()
    
