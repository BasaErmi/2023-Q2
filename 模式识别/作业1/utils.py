import cv2
import numpy as np


def harris_corners(image, block_size, ksize, k, threshold):
    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度的乘积和平方
    Ixx = grad_x**2
    Ixy = grad_x * grad_y
    Iyy = grad_y**2

    # 应用高斯滤波
    Ixx = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)

    # Harris角点响应
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    response = det - k * trace**2

    # 阈值化
    corners = response > threshold * response.max()
    corners = np.array(corners, dtype=np.uint8)

    # 对角点进行标记·
    keypoints = []
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if corners[i, j]:
                keypoints.append(cv2.KeyPoint(j, i, 1))
    return keypoints


def warp_corner(H, src):
    '''
    :param H: 单应矩阵
    :param src: 透视变化的图像
    :return: 透视变化后的四个角，左上角开始，逆时钟
    '''

    warp_points = []
    # 图像左上角，左下角
    src_left_up = np.array([0, 0, 1])
    src_left_down = np.array([0, src.shape[0], 1])

    # 图像右上角，右下角
    src_right_up = np.array([src.shape[1], 0, 1])
    src_right_down = np.array([src.shape[1], src.shape[0], 1])

    # 透视变化后的左上角，左下角
    warp_left_up = H.dot(src_left_up)
    left_up = warp_left_up[0:2] / warp_left_up[2]
    warp_points.append(left_up)
    warp_left_down = H.dot(src_left_down)
    left_down = warp_left_down[0:2] / warp_left_down[2]
    warp_points.append(left_down)

    # 透视变化后的右上角，右下角
    warp_right_up = H.dot(src_right_up)
    right_up = warp_right_up[0:2] / warp_right_up[2]
    warp_points.append(right_up)
    warp_right_down = H.dot(src_right_down)
    right_down = warp_right_down[0:2] / warp_right_down[2]
    warp_points.append(right_down)
    return warp_points


def optim_mask(mask, warp_point):
    min_left_x = min(warp_point[0][0], warp_point[1][0])
    left_margin = mask.shape[1] - min_left_x
    points_zeros = np.where(mask == 0)
    x_indexs = points_zeros[1]
    alpha = (left_margin - (x_indexs - min_left_x)) / left_margin
    mask[points_zeros] = alpha
    return mask


def Seam_Left_Right(left, imagewarp, H, warp_point, with_optim_mask=False):
    '''
    :param left: 拼接的左图像
    :param imagewarp: 透视变化后的右图像
    :param H: 单应矩阵
    :param warp_point: 透视变化后的四个顶点
    :param with_optim_mask: 是否需要对拼接后的图像进行优化
    :return:
    '''
    w = left.shape[1]
    mask = imagewarp[:, 0:w]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask != 0] = 1
    mask[mask == 0] = 0
    mask = 1 - mask
    mask = np.float32(mask)

    if with_optim_mask == True:
        mask = optim_mask(mask, warp_point)
    mask_rgb = np.stack([mask, mask, mask], axis=2)
    tt = np.uint8((1 - mask_rgb) * 255)
    left = left * mask_rgb + imagewarp[:, 0:w] * (1 - mask_rgb)
    imagewarp[:, 0:w] = left
    return np.uint8(imagewarp)
