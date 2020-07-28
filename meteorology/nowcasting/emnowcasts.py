#!/usr/bin/python3
import os
import _thread
import time
import numpy as np
import cupy as cp
import cv2
from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver
from pysteps import io, motion, nowcasts, rcparams, verification
import random
import math


def forecast(R, n_leadtimes=20, nowcastmethod='sprog', method='LK',
             n_cascade_levels=3, R_thr=0, decomp_method="fft", 
             bandpass_filter_method="gaussian", probmatching_method="cdf"):
    '''Generate a nowcast by using complementary dissipativve method.

    Parameters
    R: np.array([N, H, W]), N is history frame(gray image)

    Returns
    a np.array([M, H, W]), M is the numbe of forecast frames.
    '''
    # if all black -> reverse
    temp_R = np.where(R > 0, 1, 0)
    if np.sum(temp_R)/10 < 300:
        print('b', time.ctime(time.time()))
        temp_out = [img2dbz(i) for i in np.flip(R, 0)]
        print('e', time.ctime(time.time()))
        return temp_out + temp_out

    R_f = pysteps_method(R, n_leadtimes, nowcastmethod, method,
             n_cascade_levels, R_thr, decomp_method, 
             bandpass_filter_method, probmatching_method)
    # R_f = R
    print('b', time.ctime(time.time()))
    outs = ComplementaryDissipatives(R_f, n_leadtimes)
    print('e', time.ctime(time.time()))

    return outs

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

def pysteps_method(R, n_leadtimes, nowcastmethod, method,
                   n_cascade_levels, R_thr, decomp_method, 
                   bandpass_filter_method, probmatching_method):
    '''
    R: np.array([N, W, H])
    '''
    R, V = motion_method(R, method)
    R_f = nowcasts_method(R, V)
    return R_f

def remove_outside(src):
    h, w = src.shape[: 2]
    for i in range(h):
        for j in range(w):
            if (i-h/2)**2 + (j-w/2)**2 > (h/2)**2:
                src[i, j] = 0.0
    return src

def img2dbz(src):
    src = src.astype(np.uint8)
    if len(src.shape) == 3:
        src_grey = np.average(src, axis=2)
    else:
        src_grey = src

    src_dBz = src_grey / 2 - 32
    return src_dBz.astype(np.int8)

# ========== dissiway ==========
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

# ========== for r2c2r ==========
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

def IsOutlier(src, window=3, isouter=True):
    # h, w = src.shape[: 2]
    # if int(i*j*(i-h+1)*(j-w+1)) == 0:
    #     return False
    src[src < 10] = 0
    src = src.astype(np.uint8)
    fil = np.ones((window, window), int)
    if isouter:
        fil[int((window-1)/2)][int((window-1)/2)] = 0
    # fil = np.array([[1, 1, 1],
    #                 [1, 0, 1],
    #                 [1, 1, 1]])
    res = cv2.filter2D(src, -1, fil)

    dst = res > 10 if isouter else res[int(window/2)][int(window/2)]
    return dst

def IsContinue(src, i, j, window=5):
    h, w = src.shape[: 2]
    r = int(window/2)
    i_1 = max(i - r, 0)
    j_1 = max(j - r, 0)
    i_a1 = min(w, i + r + 1)
    j_a1 = min(h, j + r + 1)
    max_value = np.sum(src[i_1: i_a1, j_1: j_a1])
    return False if max_value > 20 else True


# ============================================

def ComplementaryDissipatives(R_f, n_leadtimes):
    return [ComplementaryDissipative(R_f[i], i) for i in range(n_leadtimes)]

def ComplementaryDissipative(src, t):
    # The first frame for optical flow
    # print('b', t, time.ctime(time.time()))
    dst = remove_outside(src)
    if t < 1:
        # print('e', t, time.ctime(time.time()))
        return img2dbz(dst)
    dst = DissipativeOpticalFlow(dst, t)
    # print('e', t, time.ctime(time.time()))
    return dst

def r2c2r(i, j, center_x, center_y, thetas, rs):
    # r r2c
    x_hat, y_hat = i - center_x, j - center_y
    r = int(np.sqrt(x_hat*x_hat+y_hat*y_hat))
    if r > center_x:
        return i, j
    r_idx = FindBestR(r, rs)
    r_best = rs[r_idx]
    theta_idx = FindBestTheta(x_hat, y_hat, thetas)
    theta_best = thetas[theta_idx]
    # r c2r
    x = int(r_best*np.cos(theta_best) + center_x)
    y = int(r_best*np.sin(theta_best) + center_y)
    return x, y

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

def DissipativeOpticalFlow(src, t, rmax=360, thetamax=720):
    src_dbz = img2dbz(src)
    h, w = src.shape[: 2]
    dst = np.copy(src_dbz)
    iscontinue = np.zeros((h, w))
    center_x, center_y = [int(i/2) for i in src.shape[: 2]]
    thetas = np.linspace(0, 2*np.pi, thetamax)
    rs = np.linspace(0, center_x-1, rmax)
    for i in range(int(h)):
        for j in range(int(w)):
            if IsContinue(src, i, j, 3):
                continue
            x, y = r2c2r(i, j, center_x, center_y, thetas, rs)
            # dissipative
            if iscontinue[x][y] == 0:
                dbz = src_dbz[x][y]
                dissipative_dbz = DissipativeRun(dbz, t)
                iscontinue[x][y] = dissipative_dbz
            else:
                dissipative_dbz = iscontinue[x][y]
            dst[i][j] = dissipative_dbz
    res = dst*IsOutlier(dst)
    # res = res.astype(np.uint8)
    # remycire = cv2.resize(res, (360, 360), interpolation=cv2.INTER_CUBIC)
    # mycire = cv2.resize(remycire, (720, 720), interpolation=cv2.INTER_CUBIC)
    # res = dst
    return res        
