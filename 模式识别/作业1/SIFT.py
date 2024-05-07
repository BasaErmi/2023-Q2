import cv2
import numpy as np
from utils import harris_corners

# 读取图像
image1 = cv2.imread('images/uttower1.jpg')
image2 = cv2.imread('images/uttower2.jpg')

# 使用SIFT特征描述子获取关键点的特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

print(descriptors1.shape)
print(descriptors2.shape)

# 创建BFMatcher对象并显式指定距离度量方式为欧几里得距离
bf = cv2.BFMatcher(normType=cv2.NORM_L2)

# 使用欧几里得距离进行匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比值测试
good_matches = []
for m, n in matches:
    if m.distance < 0.60 * n.distance:
        good_matches.append(m)

# Draw matches
image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 保存匹配结果
cv2.imwrite('results/uttower_match_sift.png', image_matches)