import time
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import cupy as cp

# print(cv2.getBuildInformation())

def ExpTran_matrix(img, esp=0, gama=1):
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gamma_trans = img/255
    gamma_trans = np.power(gamma_trans+esp,gama)*255
    gamma_trans = np.asarray(gamma_trans, dtype=np.int)
    gamma_trans[gamma_trans<0]=0
    gamma_trans[gamma_trans>255]=255
    return gamma_trans

def sparse_flow(flow, X=None, Y=None, stride=1):
    flow = flow.copy()
    flow[:, :, 0] = -flow[:, :, 0]
    if X is None:
        height, width, _ = flow.shape
        xx = np.arange(0, height, stride)
        yy = np.arange(0, width, stride)
        X, Y = np.meshgrid(xx, yy)
        X = X.flatten()
        Y = Y.flatten()

        # sample
        sample_0 = flow[:, :, 0][xx]
        sample_0 = sample_0.T
        sample_x = sample_0[yy]
        sample_x = sample_x.T
        sample_1 = flow[:, :, 1][xx]
        sample_1 = sample_1.T
        sample_y = sample_1[yy]
        sample_y = sample_y.T

        sample_x = sample_x[:, :, np.newaxis]
        sample_y = sample_y[:, :, np.newaxis]
        new_flow = np.concatenate([sample_x, sample_y], axis=2)
    flow_x = new_flow[:, :, 0].flatten()
    flow_y = new_flow[:, :, 1].flatten()

    # display
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    # plt.quiver(X,Y, flow_x, flow_y, angles="xy", color="#666666")
    ax.quiver(X, Y, flow_x, flow_y, color="#666666")
    ax.grid()
    # ax.legend()
    plt.draw()
    plt.show()

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v, colorwheel):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = np.size(colorwheel, 0)    # 55 colorwheel: (55, 3)

    rad = np.sqrt(u**2+v**2)          # (h, w)

    a = np.arctan2(-v, -u) / np.pi    # (h, w)

    fk = (a+1) / 2 * (ncols - 1) + 1  # (h, w)

    k0 = np.floor(fk).astype(int)     # (h, w)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    # s1 = time.time()
    for i in range(0, np.size(colorwheel,1)):   # 3
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    # e1 = time.time()
    # print(e1 - s1)

    return img

def compute_color_cp(u, v, colorwheel):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = cp.zeros((h, w, 3))
    nanIdx = cp.isnan(u) | cp.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    ncols = 55    # 55 colorwheel: (55, 3)

    rad = cp.sqrt(u**2+v**2)          # (h, w)

    a = cp.arctan2(-v, -u) / cp.pi    # (h, w)

    fk = (a+1) / 2 * (ncols - 1) + 1  # (h, w)

    k0 = cp.floor(fk).astype(int)     # (h, w)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    # s1 = time.time()
    for i in range(0, 3):   # 3
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = cp.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = cp.floor(255 * col*(1-nanIdx)).astype(cp.uint8)
    # e1 = time.time()
    # print(e1 - s1)

    return img

def flow_to_image(flow, colorwheel):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # maxu = -999.
    # maxv = -999.
    # minu = 999.
    # minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    # SMALLFLOW = 0.0
    # LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # maxu = max(maxu, np.max(u))
    # minu = min(minu, np.min(u))
    #
    # maxv = max(maxv, np.max(v))
    # minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))            #   78.27949

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)


    img = compute_color(u, v, colorwheel)   # 最耗时间



    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def flow_to_image_cp(flow, colorwheel):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    flow = cp.asarray(flow)
    colorwheel = cp.asarray(colorwheel)

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # maxu = -999.
    # maxv = -999.
    # minu = 999.
    # minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    # SMALLFLOW = 0.0
    # LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # maxu = max(maxu, np.max(u))
    # minu = min(minu, np.min(u))
    #
    # maxv = max(maxv, np.max(v))
    # minv = min(minv, np.min(v))

    rad = cp.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, cp.max(rad))             # 78.27949

    u = u/(maxrad + cp.finfo(float).eps)
    v = v/(maxrad + cp.finfo(float).eps)


    img = compute_color_cp(u, v, colorwheel)   # 最耗时间

    idx = cp.repeat(idxUnknow[:, :, cp.newaxis], 3, axis=2)
    img[idx] = 0

    img = cp.asnumpy(img)

    return img.astype(np.uint8)

def dense_OF(cur,pre):
    """
    :param cur:  np.uint8
    :param pre:
    :return:
    """

    colorwheel = make_color_wheel()

    next = cur
    old_gray = pre

    """
    Farneback
    """
    flow = cv2.calcOpticalFlowFarneback(prev=old_gray,next=next, flow=None, pyr_scale=0.5, levels=3, winsize=7, iterations=9, poly_n=7, poly_sigma=1.5, flags=0)  # (prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    """
    Farneback_cuda
    """
    # gpu_frame_old = cv2.cuda_GpuMat()
    # gpu_frame_old.upload(old_gray)
    #
    # gpu_frame_next = cv2.cuda_GpuMat()
    # gpu_frame_next.upload(next)
    # optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
    # gpu_flow = optical_flow.calc(gpu_frame_old, gpu_frame_next, None)
    # flow = gpu_flow.download()
    """
    TV_L1, CPU
    """
    # optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(nscales=5,epsilon=0.01,warps=5)
    # flow = optical_flow.calc(old_gray, next, None)

    """
    cuda TV-L1
    """
    ## optical_flow = cv2.cuda_NvidiaOpticalFlow_1_0.create((800, 600))

    # gpu_frame_old = cv2.cuda_GpuMat()
    # gpu_frame_old.upload(old_gray)
    #
    # gpu_frame_next = cv2.cuda_GpuMat()
    # gpu_frame_next.upload(next)
    #
    # optical_flow = cv2.cuda_OpticalFlowDual_TVL1.create(nscales=5,epsilon=0.01,warps=5)
    # # optical_flow = cv2.cuda_FarnebackOpticalFlow.create()
    # gpu_flow = optical_flow.calc(gpu_frame_old, gpu_frame_next, None)
    # flow = gpu_flow.download()

    """
    accumulate
    """
    # for j in range(seq_len - 1):
    #     dense_optical_flow_volume[j] = dense_optical_flow_volume[j + 1]
    # dense_optical_flow_volume[seq_len - 1] = flow
    #
    # mean_dense_flow = np.mean(dense_optical_flow_volume,axis=0)


    # sparse_flow(flow,stride=10)
    rgb = flow_to_image(flow, colorwheel)

    return rgb


