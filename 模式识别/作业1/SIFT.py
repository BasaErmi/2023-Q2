import cv2
import numpy as np
from utils import warp_corner, optim_mask, Seam_Left_Right, harris_corners

# 读取图像
image1 = cv2.imread('results/yosemite5.jpg')
image2 = cv2.imread('results/yosemite6.jpg')

# 使用SIFT特征描述子获取关键点的特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


# 创建BFMatcher对象并指定度量方式为欧几里得距离
bf = cv2.BFMatcher(normType=cv2.NORM_L2)

# 使用knn临近匹配算法
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比值测试
good_matches = []
for m, n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 保存匹配结果
cv2.imwrite('results/uttower_match_sift.png', image_matches)

# 获取匹配点的坐标
left_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
right_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# 使用RANSAC算法求解仿射变换矩阵
H, _ = cv2.findHomography(right_points, left_points, method=cv2.RANSAC, ransacReprojThreshold=5)

# 求出右图像的透视变化顶点
warp_point = warp_corner(H, image2)

# 求出右图像的透视变化图像
imagewarp = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image2.shape[0]))

# 对左右图像进行拼接
result = Seam_Left_Right(image1, imagewarp, H, warp_point, with_optim_mask=False)

cv2.imshow('Result', result)
cv2.waitKey(0)

# 保存结果
cv2.imwrite('results/yosemite_stitching.png', result)
