import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corners(image, block_size, ksize, k, threshold):
    # 灰度化图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # 计算梯度的乘积和平方
    Ixx = grad_x ** 2
    Ixy = grad_x * grad_y
    Iyy = grad_y ** 2

    # 应用高斯滤波
    A = cv2.GaussianBlur(Ixx, (block_size, block_size), 0)
    B = cv2.GaussianBlur(Ixy, (block_size, block_size), 0)
    C = cv2.GaussianBlur(Iyy, (block_size, block_size), 0)

    # Harris角点响应
    det = A * C - B ** 2
    trace = A + C
    response = det - k * trace ** 2

    # 阈值化
    corners = response > threshold * response.max()
    corners = np.array(corners, dtype=np.uint8)

    # 对角点进行标记
    result = image.copy()
    result[corners > 0] = [0, 0, 255]

    return result


# 读取图像
image = cv2.imread('images/sudoku.png')

# 应用Harris角点检测
result_image = harris_corners(image, block_size=3, ksize=3, k=0.04, threshold=0.01)
# 显示结果
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# 保存结果
cv2.imwrite('results/sudoku_keypoints.jpg', result_image)

# 读取图像
image1 = cv2.imread('images/uttower1.jpg')
image2 = cv2.imread('images/uttower2.jpg')

# 提取关键点
result_image1 = harris_corners(image1, block_size=3, ksize=3, k=0.04, threshold=0.01)
result_image2 = harris_corners(image2, block_size=3, ksize=3, k=0.04, threshold=0.01)

# 保存结果
cv2.imwrite('results/uttower1_keypoints.jpg', result_image1)
cv2.imwrite('results/uttower2_keypoints.jpg', result_image2)