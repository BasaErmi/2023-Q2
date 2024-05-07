import cv2
import numpy as np

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


# 读取图像
image1 = cv2.imread('images/uttower1.jpg')
image2 = cv2.imread('images/uttower2.jpg')

# 使用SIFT特征描述子获取关键点的特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 创建BFMatcher对象，使用欧几里得距离进行特征匹配
bf = cv2.BFMatcher(normType=cv2.NORM_L2)

# 利用knn对左右图像的特征点进行匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比值测试
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 获取匹配点的坐标
left_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
right_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# 使用RANSAC算法求解仿射变换矩阵
H, _ = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5)

# 求出右图像的透视变化顶点
warp_point = warp_corner(H, image2)

# 求出右图像的透视变化图像
imagewarp = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image2.shape[0]))

# 对左右图像进行拼接
result = Seam_Left_Right(image1, imagewarp, H, warp_point, with_optim_mask=True)

cv2.imshow('Result', result)
cv2.waitKey(0)

# 保存结果
cv2.imwrite('results/uttower_stitching_sift.png', result)
