import cv2
import numpy as np

# Harris角点检测
def harris_corners(image, block_size, ksize, k, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    Ixx = grad_x**2
    Ixy = grad_x * grad_y
    Iyy = grad_y**2
    Ixx = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    response = det - k * trace**2
    corners = response > threshold * response.max()
    corners = np.array(corners, dtype=np.uint8)
    keypoints = []
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if corners[i, j]:
                keypoints.append(cv2.KeyPoint(j, i, 1))
    return keypoints

# 读取图像
image1 = cv2.imread('images/uttower1.jpg')
image2 = cv2.imread('images/uttower2.jpg')

# 提取关键点
keypoints1 = harris_corners(image1, block_size=3, ksize=3, k=0.04, threshold=0.01)
keypoints2 = harris_corners(image2, block_size=3, ksize=3, k=0.04, threshold=0.01)

# 绘制关键点并保存结果
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imwrite('results/uttower1_keypoints.jpg', image1_with_keypoints)
cv2.imwrite('results/uttower2_keypoints.jpg', image2_with_keypoints)

# 使用SIFT特征描述子获取关键点的特征
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

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

