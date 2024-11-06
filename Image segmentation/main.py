import cv2
import numpy as np
import os
import glob
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# 读取图像
directory_path = 'regional'

# 支持的图片格式
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

# 获取所有图片文件的完整路径
image_files = []
for extension in image_extensions:
    image_files.extend(glob.glob(os.path.join(directory_path, extension)))

# 遍历每个图像文件，读取并转换为灰度图像
for image_path in image_files:
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)

    # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 初始化变量以存储边界直线
    top_line = None
    bottom_line = None
    left_line = None
    right_line = None

    # 定义一个函数来计算斜率
    def calculate_slope(x1, y1, x2, y2):
        if x1 == x2:
            return float('inf')  # 垂直线
        else:
            return (y2 - y1) / (x2 - x1)

    # 筛选直线
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = calculate_slope(x1, y1, x2, y2)

            # 接近水平的线（斜率接近0）
            if abs(slope) < 0.1:
                if top_line is None or y1 < top_line[1]:
                    top_line = (x1, y1, x2, y2)
                if bottom_line is None or y1 > bottom_line[1]:
                    bottom_line = (x1, y1, x2, y2)

            # 接近垂直的线（斜率很大）
            elif abs(slope) > 10:
                if left_line is None or x1 < left_line[0]:
                    left_line = (x1, y1, x2, y2)
                if right_line is None or x1 > right_line[0]:
                    right_line = (x1, y1, x2, y2)

    # 确保找到了所有边界线
    if top_line and bottom_line and left_line and right_line:
        # 计算交点
        def intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # 平行线无交点

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return int(px), int(py)

        # 计算四个交点
        top_left = intersection(top_line, left_line)
        top_right = intersection(top_line, right_line)
        bottom_left = intersection(bottom_line, left_line)
        bottom_right = intersection(bottom_line, right_line)

        if None not in (top_left, top_right, bottom_left, bottom_right):
            min_x = min(top_left[0], bottom_left[0])
            max_x = max(top_right[0], bottom_right[0])
            min_y = min(top_left[1], top_right[1])
            max_y = max(bottom_left[1], bottom_right[1])

            cropped_image = image[min_y:max_y, min_x:max_x]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            # 保存裁剪后的图像
            save_path = f'{base_name}_cropped.jpg'
            cv2.imwrite(save_path, cropped_image)
            print('succeed')
    else:
        print("failed")