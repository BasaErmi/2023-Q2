import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift_match(img1, img2, output_path):
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 使用SIFT找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建BFMatcher对象，使用欧式距离
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制前10个匹配
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配结果图像
    plt.imshow(img_matches)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


# 加载图像
img1 = cv2.imread('img/uttower1.jpg')
img2 = cv2.imread('img/uttower2.jpg')

# 转换为RGB格式
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 运行SIFT匹配并保存结果
sift_match(img1, img2, 'results/uttower_match_sift.png')
