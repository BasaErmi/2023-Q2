import cv2
import numpy as np

# 读取图像
image1 = cv2.imread('images/uttower1.jpg')
image2 = cv2.imread('images/uttower2.jpg')

# 使用SIFT特征描述子获取关键点的特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher()

# 使用欧几里得距离进行匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 获取匹配点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法估计仿射变换矩阵
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

# 对图像1进行仿射变换
rows, cols = image1.shape[:2]
image1_warped = cv2.warpAffine(image1, M, (cols, rows))

# 将两个图像拼接在一起
result_image = cv2.hconcat([image1_warped, image2])

# 保存拼接结果
cv2.imwrite('results/uttower_stitching_sift.png', result_image)
