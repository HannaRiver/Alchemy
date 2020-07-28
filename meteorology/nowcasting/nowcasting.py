#!/usr/bin/python3
import os
import _thread
import time
import numpy as np
import cv2
from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver
from pysteps import io, motion, nowcasts, rcparams, verification
# from numba import jit
import random
import math


def readTxt(txtpath):
    imgs = np.zeros((10, 720, 720))
    with open(txtpath, 'r') as f:
        for i, line in enumerate(f.readlines()):
            img = cv2.imread(line.strip(), 0)
            imgs[i] = img
    return imgs

def dbz2colormap(src_dbz, bgr=True):
    src_color = img_color(src_dbz, bgr)
    if bgr:
        src_color = np.array(src_color, dtype=np.uint8)
        src_color = cv2.cvtColor(src_color, cv2.COLOR_RGB2BGR)
    else:
        src_color = np.array(src_color, dtype=np.int8)
    return src_color

def img_color(img, bgr=True, heat=False):
    color_translate = [(0, 0, 0), (1, 159, 246), (0, 236, 236), (1, 216, 0), (1, 144, 0),
                       (255, 255, 0), (231, 192, 0), (255, 144, 0),
                       (254, 0, 0), (214, 0, 0), (192, 0, 0),
                       (255, 0, 240), (149, 0, 180), (174, 144, 240)]  # RGB

    if heat:
        color_translate = [(0, 0, 0), (0, 0, 255), (0, 84, 255), (0, 168, 255), (0, 255, 255),
                           (0, 255, 168), (0, 255, 84), (0, 255, 0),
                           (84, 255, 0), (168, 255, 0), (255, 255, 0),
                           (255, 168, 0), (255, 84, 0), (255, 0, 0)]  # RGB
    if not bgr:
        #  color_translate = [i for i, j in enumerate(color_translate)]
        color_translate = [range(len(color_translate))]

    cl = len(color_translate)

    mm = np.array(color_translate)
    ind = (img / 5).astype('int')
    ind -= 1
    ind = np.clip(ind, a_min=0, a_max=cl - 1)

    res = mm[ind]
    return res

def dbz2img(src):
    src[src<0] = 0
    dst = (src + 32) * 2
    return dst

def remove_outside(src):
    h, w = src.shape[: 2]
    for i in range(h):
        for j in range(w):
            if (i-h/2)**2 + (j-w/2)**2 > (h/2)**2:
                src[i, j] = 0.0
    return src

def img_to_dbz(src):
    src = src.astype(np.uint8)
    if len(src.shape) == 3:
        src_grey = np.average(src, axis=2)
    else:
        src_grey = src

    src_dBz = src_grey / 2 - 32
    return src_dBz

def nowcasts_of(nowcastmethod='sprog'):
    '''
    Nowcasts for Optical flow
    '''
    pass

def motion_method(R, method='LK'):
    oflow_method = motion.get_method(method)
    motion.get_method(method)
    V = oflow_method(R)
    return R, V 

def nowcasts_method(R, V, n_leadtimes=20, nowcastmethod='sprog'):
    nowcast_method = nowcasts.get_method(nowcastmethod)
    R_f = nowcast_method(R, V, n_leadtimes, n_cascade_levels=3,
            R_thr=0, decomp_method="fft", bandpass_filter_method="gaussian",
            probmatching_method="cdf")
    return R_f

def pysteps_method(R):
    '''
    R: np.array([N, W, H])
    '''
    R, V = motion_method(R)
    R_f = nowcasts_method(R, V)
    return R_f

def dBz2Z(dbz):
    return math.pow(10, dbz/10)

def Z2dBz(Z):
    return 10*math.log10(Z) if Z > 0 else -32

def dissiwayFuncSample(dbz, i):
    if dbz < 10:
        return dbz
    diff_list = np.array([15, 50, 100, 300, 1000, 2000, 5000, 10000, 300000, 100000, 300000, 1000000])/6
    diff_idx = int((dbz - 10)/5)
    diff_idx = diff_idx if diff_idx <= 11 else -1
    Z = dBz2Z(dbz)
    func_list = [Z - diff_list[diff_idx], Z + diff_list[diff_idx], Z - 2 * diff_list[diff_idx]]
    dbz = Z2dBz(func_list[i])
    return dbz

def dissiwayFunc(dbz, t):
    if dbz < 10:
        # return dbz if dbz > 10 else 0
        return dbz
    random_val = random.random()
    prob_list = [0.1, 0.4, 0.5, 0.1] # +1, 0, -1, -2
    if random_val < prob_list[0]:
        dbz = dissiwayFuncSample(dbz, 1)
        return dbz
    elif random_val < sum(prob_list[: 2]):
        return dbz
    elif random_val < sum(prob_list[: 3]):
        dbz = dissiwayFuncSample(dbz, 0)
        return dbz
    elif random_val > prob_list[-1]:
        dbz = dissiwayFuncSample(dbz, 2)
        return dbz
    return dbz
def DissipativeRun(dbz, t):
    if t < 1:
        return dbz
    dbz = dissiwayFunc(dbz, t)
    t -= 1
    return DissipativeRun(dbz, t)

def CalTheta(x_hat, y_hat):
    if x_hat == 0 and y_hat == 0:
        return 0
    elif x_hat == 0:
        cal_theta = np.pi/2 if y_hat > 0 else 3*np.pi/2
        return cal_theta
    else:
        cal_theta = np.arctan(y_hat/x_hat)
    if x_hat <= 0 and y_hat <= 0:
        cal_theta = cal_theta if cal_theta >= np.pi else cal_theta + np.pi
    elif x_hat < 0 and y_hat > 0:
        # cal_theta = cal_theta if cal_theta < np.pi else cal_theta - np.pi
        cal_theta = cal_theta + np.pi
    elif x_hat > 0 and y_hat > 0:
        cal_theta = cal_theta if cal_theta < np.pi else cal_theta - np.pi
    elif x_hat > 0 and y_hat < 0:
        # cal_theta = cal_theta if cal_theta > np.pi else cal_theta + np.pi
        cal_theta = cal_theta + 2*np.pi
        
    return  cal_theta

def FindBestTheta(x_hat, y_hat, thetas):
    cal_theta = CalTheta(x_hat, y_hat)
    distancs = np.array(thetas) - cal_theta
    distancs_sq = distancs*distancs
    best_idx = np.argmin(distancs_sq)
    return best_idx

def FindBestR(r, rs):
    distancs = np.array(rs) - r
    distancs_sq = distancs*distancs
    best_idx = np.argmin(distancs_sq)
    return best_idx

#@jit
def DissipativeOpticalFlow(dbz_frame, t=0, windows=3):
    if t < 1:
        return dbz_frame
    h, w = dbz_frame.shape
    for i in range(int(h)):
        for j in range(int(w)):
            # x, y = int(windows*i+(windows-1)/2), int(windows*j+(windows-1)/2)
            x, y = i, j
            dbz = dbz_frame[x][y]
            dbz_frame[x][y] = DissipativeRun(dbz, t)
    return dbz_frame

#@jit
def Cirecle2Rectangle(src, rmax=720, thetamax=720):
    center_x, center_y = [int(i/2) for i in src.shape]
    dst = np.zeros((rmax, thetamax))
    thetas = np.linspace(0, 2*np.pi, thetamax)
    rs = np.linspace(0, center_x-1, rmax)
    for i, theta in enumerate(thetas):
        for j, r in enumerate(rs):
            # 这里的int是十分粗糙的
            x = int(r*np.cos(theta) + center_x)
            y = int(r*np.sin(theta) + center_y)
            dst[j, i] = src[x, y]
    return dst

#@jit
def Rectangle2Cirecle(src, centers=(360, 360)):
    center_x, center_y = centers
    dst = np.zeros((2*center_x, 2*center_y))
    max_theta = src.shape[1]
    max_r = src.shape[0]
    thetas = np.linspace(0, 2*np.pi, max_theta)
    rs = np.linspace(0, center_x-1, max_r)
    for x in range(2*center_x):
        for y in range(2*center_y):
            x_hat = x - center_x
            y_hat = y - center_y
            r = int(np.sqrt(x_hat*x_hat+y_hat*y_hat))
            r_idx = FindBestR(r, rs)
            theta_idx = FindBestTheta(x_hat, y_hat, thetas)
            dst[x, y] = src[r_idx, theta_idx] if r < 360 else 0
    return dst

def buity4cire(dissi_frame):
    mycire = Cirecle2Rectangle(dissi_frame)
    remycire = cv2.resize(mycire, (360, 360), interpolation=cv2.INTER_CUBIC)
    mycire = cv2.resize(remycire, (720, 720), interpolation=cv2.INTER_CUBIC)
    my_img = Rectangle2Cirecle(mycire)
    return my_img

def ComplementaryDissipative(src, t=0, save_dir=''):
    img_dbz = img_to_dbz(src)
    dissi_frame = DissipativeOpticalFlow(img_dbz, t, 1)
    # dissi_frame = dissi_frame*IsOutlier(dissi_frame)
    if t < 1:
        # print(t,'done', time.ctime(time.time()))
        # cv2.imwrite(os.path.join(save_dir, str(t)+'.png'),  dbz2colormap(dissi_frame))
        return remove_outside(dissi_frame)
    finecire = buity4cire(dissi_frame)
    finecire = finecire * IsOutlier(finecire)
    # gray_frame = dbz2img(finecire)
    # gray_frame = dbz2colormap(finecire)
    # print(t,'done', time.ctime(time.time()))
    return finecire
    # cv2.imwrite(os.path.join(save_dir, str(t)+'.png'), gray_frame)

def c2r2c(R_f, save_dir):
    out = [ComplementaryDissipative(R_f[i], i, save_dir) for i in range(20)]
    return out
        # _thread.start_new_thread(ComplementaryDissipative, (R_f[i], i, save_dir))

def IsOutlier(src, window=3):
    # h, w = src.shape[: 2]
    # if int(i*j*(i-h+1)*(j-w+1)) == 0:
    #     return False
    src[src < 10] = 0
    src = src.astype(np.uint8)
    fil = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
    res = cv2.filter2D(src, -1, fil)

    dst = res > 10
    return dst

def forecast(R):
    R_f = pysteps_method(R)
    # R_f = R
    return c2r2c(R_f, '')

def Test4c2r2c(R):
    img_lists = '/work/meteorology/data/radar/HongQiao/cappi/gt/test.txt'
    save_dir = '/work/meteorology/data/radar/HongQiao/cappi/exp/c2r2c2'
    # R = readTxt(img_lists)
    # print('done', time.ctime(time.time()))
    R_f = pysteps_method(R)
    print('done', time.ctime(time.time()))
    # R_f[np.isnan(R_f)] = 0
    return c2r2c(R_f, save_dir)

if __name__ == '__main__':
    # R = ''
    # R_f = c2r2c(R)
    Test4c2r2c()