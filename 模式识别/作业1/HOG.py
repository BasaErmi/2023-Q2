import cv2
import numpy as np

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
image1 = cv2.imread('images/uttower1.jpg') # image1.shape = (410, 615, 3)
image2 = cv2.imread('images/uttower2.jpg')

# 提取关键点
keypoints1 = harris_corners(image1, block_size=3, ksize=3, k=0.04, threshold=0.01)
keypoints2 = harris_corners(image2, block_size=3, ksize=3, k=0.04, threshold=0.01)

print(f"Number of keypoints in image 1: {len(keypoints1)}")
print(f"Number of keypoints in image 2: {len(keypoints2)}")
# for i in range(8500, 8510):
#     print(keypoints1[i].pt)

# 将坐标变为整数
locations1 = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints1]
locations2 = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints2]

# for i in range(10):
#     print(locations1[i])
#     print(type(locations1[i]))

# 对每个关键点周围的区域计算HOG描述子
hog = cv2.HOGDescriptor()
descriptors1 = hog.compute(image1, locations=locations1[:10])
descriptors2 = hog.compute(image2, locations=locations2[:10])
 b
print(f"shape of descriptors1: {descriptors1.shape}")
print(f"shape of descriptors2: {descriptors2.shape}")


# 创建BFMatcher对象并显式指定距离度量方式为欧几里得距离
bf = cv2.BFMatcher(normType=cv2.NORM_L2)

# 使用欧几里得距离进行暴力匹配
matches = bf.match(descriptors1, descriptors2)

# 按照距离进行排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制匹配结果
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()