#/usr/bin/python3
#-*- encoding=utf-8 -*-

import os
import sys
tool_root = os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]), '..', '..', '..'))
sys.path.append(os.path.join(tool_root, 'utils'))

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

from resize_img import batch_resize_img


def mk_ca_resize_data(imgtxt, save_dir):
    version = "V0.1.2"
    mean_value = [150, 139, 138]
    resize_h, resize_w = 48, 240

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        logging.info("Save path not exists, make it!")

    logging.info("==================== Begin make resize data for Uppercase date ====================")
    logging.info("Parameter Version: %s" %(version))
    logging.info("Mean Value: %s" %(str(mean_value)))
    logging.info("Resize Height: %s \t Resize Width: %s" %(resize_h, resize_w))

    batch_resize_img(imgtxt, resize_h, resize_w, mean_value, save_dir)

    logging.info("-------------------------------------- Done ---------------------------------------")


def print_help():
    print("""Usage:
        ./mk_cd_resize_data.py /imglist/to/resize.txt /dir/to/save/resized data ...
    """)
    sys.exit()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print_help()
    imgtxt = sys.argv[1]
    save_dir = sys.argv[2]
    mk_ca_resize_data(imgtxt, save_dir)
