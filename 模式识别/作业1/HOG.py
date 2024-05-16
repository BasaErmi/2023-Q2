import cv2
import numpy as np
from utils import harris_corners, warp_corner, optim_mask, Seam_Left_Right

# 读取图像
image1 = cv2.imread('images/uttower1.jpg') # image1.shape = (410, 615, 3)
image2 = cv2.imread('images/uttower2.jpg')

# 使用Harris角点检测算法提取关键点
# keypoints1 = harris_corners(image1, block_size=3, ksize=3, k=0.04, threshold=0.01)
# keypoints2 = harris_corners(image2, block_size=3, ksize=3, k=0.04, threshold=0.01)

# 使用SIFT提取关键点
sift = cv2.SIFT_create()
keypoints1, _ = sift.detectAndCompute(image1, None)
keypoints2, _ = sift.detectAndCompute(image2, None)

print(f"len of keypoints1: {len(keypoints1)}")
print(f"len of keypoints2: {len(keypoints2)}")

# 将坐标变为整数
locations1 = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints1]
locations2 = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints2]

# for i in range(10):
#     print(locations1[i])
#     print(type(locations1[i]))

# 对每个关键点周围的区域计算HOG描述子
hog = cv2.HOGDescriptor()
descriptors1 = hog.compute(image1, locations=locations1)
descriptors2 = hog.compute(image2, locations=locations2)

print(f"shape of descriptors1: {descriptors1.shape}")

descriptors1 = descriptors1.reshape(len(keypoints1), -1)
descriptors2 = descriptors2.reshape(len(keypoints2), -1)
print(f"shape of descriptors1: {descriptors1.shape}")
print(f"shape of descriptors2: {descriptors2.shape}")


# 创建BFMatcher对象并显式指定距离度量方式为欧几里得距离
bf = cv2.BFMatcher(normType=cv2.NORM_L2)

# 使用欧几里得距离进行暴力匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 使用比值测试获取好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.94 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('results/uttower_match_hog.png', result)

# 对图像进行拼接
left_points = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
right_points = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

# 使用RANSAC算法求解仿射变换矩阵
H, _ = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5)

warp_point = warp_corner(H, image2)
imagewarp = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image2.shape[0]))

result = Seam_Left_Right(image1, imagewarp, H, warp_point, with_optim_mask=True)

cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('results/uttower_stitching_hog.png', result)
