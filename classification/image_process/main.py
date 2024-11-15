#%%
import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import random
import os
import glob
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())
#%%

def adjust_color_temperature(im, factor):
    """调整图像色温

    Args:
        im (三维): 原始图像数据
        factor (数值): 调节冷暖数值（<1为冷，>1为暖）

    Returns:
        三维: 输出图像数据
    """
    r, g, b = im.split()
    r_enhancer = ImageEnhance.Brightness(r)
    b_enhancer = ImageEnhance.Brightness(b)
    if factor > 1:
        r = r_enhancer.enhance(factor)
        b = b_enhancer.enhance(1 / factor)
    else:
        r = r_enhancer.enhance(1 / factor)
        b = b_enhancer.enhance(factor)

    return Image.merge('RGB', (r, g, b))

def simulate_exposure(im, center, radius, intensity):
    """以一点为中心，进行模拟过曝

    Args:
        im （三维）: 图像数据
        center （三维）: 中心点
        radius （数值）: 曝光范围
        intensity （数值）: 曝光强度

    Returns:
        图像数据: 图像数据（三维）
    """
    np_im = np.array(im, dtype=np.float32)
    height, width, _ = np_im.shape
    y, x = np.ogrid[:height, : width]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = np.clip((radius - distance) / radius, 0, 1) * intensity
    for i in range(3):
        np_im[..., i] += mask * 255
    np.im = np.clip(np_im, 0, 255).astype(np.uint8)
    return Image.fromarray(np.im)

def random_crop(image, max_crop_ratio=0.1):
    """按一定概率裁剪四周

    Args:
        image (三维): 原始图像数据
        max_crop_ratio (float, optional): 最大裁剪范围（0-1）. Defaults to 0.1.

    Returns:
        三维: 结果图像数据
    """
    width, height = image.size
    # 计算最大裁剪像素数
    max_crop_width = int(width * max_crop_ratio)
    max_crop_height = int(height * max_crop_ratio)
    left_crop = 0
    right_crop = 0
    top_crop = 0
    bottom_crop = 0
    if random.betavariate(2,2) > 0.7:
        left_crop = random.randint(0, max_crop_width)
    if random.betavariate(2,2) > 0.7:
        right_crop = random.randint(0, max_crop_width)
    if random.betavariate(2,2) > 0.7:
        top_crop = random.randint(0, max_crop_height)
    if random.betavariate(2,2) > 0.7:
        bottom_crop = random.randint(0, max_crop_height)

    left = left_crop
    right = width - right_crop
    top = top_crop
    bottom = height - bottom_crop

    return image.crop((left, top, right, bottom))

def process_image(path, size, rotate, num):
    im = Image.open(path)
    im = im.resize((size,size))
    base_name = os.path.splitext(os.path.basename(path))[0]
    output_dir = os.path.join(os.path.dirname(path), base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num):
        pic = im
        if random.betavariate(2,2) > 0.4:
            pic = ImageEnhance.Contrast(pic).enhance(random.gauss(1,0.15))#对比度
        if random.betavariate(2,2) > 0.4:
            pic = adjust_color_temperature(pic, random.gauss(1, 0.15))#色温
        if random.betavariate(2,2) > 0.4:
            pic = simulate_exposure(pic, center=(random.uniform(0,size),random.uniform(0,size)), radius=random.uniform(0,size/2), intensity=random.gauss(0.5, 0.1))#过曝
        if random.betavariate(2,2) > 0.4:
            pic = random_crop(pic, max_crop_ratio=0.2)#裁剪
        if random.betavariate(2,2) > 0.4:
            pic = pic.rotate(random.uniform(-15,15))#旋转
        output_path = os.path.join(output_dir, f"{base_name}{i}.png")
        pic.save(output_path)
        
        
current_dir = os.getcwd()
jpg_files = glob.glob(os.path.join(current_dir, "*.jpg"))
rotate = 15
size = 500
num = 100
for path in tqdm.tqdm (jpg_files, desc = 'Processing...',
                  ascii = False, ncols = 75):
    process_image(path,size,rotate,num)

# %%
# import numpy as np
# import imgaug as ia
# import imgaug.augmenters as iaa


# ia.seed(1)

# # Example batch of images.
# # The array has shape (32, 64, 64, 3) and dtype uint8.
# images = np.array(
#     [ia.quokka(size=(64, 64)) for _ in range(32)],
#     dtype=np.uint8
# )

# seq = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontal flips
#     iaa.Crop(percent=(0, 0.1)), # random crops
#     # Small gaussian blur with random sigma between 0 and 0.5.
#     # But we only blur about 50% of all images.
#     iaa.Sometimes(
#         0.5,
#         iaa.GaussianBlur(sigma=(0, 0.5))
#     ),
#     # Strengthen or weaken the contrast in each image.
#     iaa.LinearContrast((0.75, 1.5)),
#     # Add gaussian noise.
#     # For 50% of all images, we sample the noise once per pixel.
#     # For the other 50% of all images, we sample the noise per pixel AND
#     # channel. This can change the color (not only brightness) of the
#     # pixels.
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#     # Make some images brighter and some darker.
#     # In 20% of all cases, we sample the multiplier once per channel,
#     # which can end up changing the color of the images.
#     iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     # Apply affine transformations to each image.
#     # Scale/zoom them, translate/move them, rotate them and shear them.
#     iaa.Affine(
#         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
#         rotate=(-25, 25),
#         shear=(-8, 8)
#     )
# ], random_order=True) # apply augmenters in random order

# images_aug = seq(images=images)
# %%
# import random  
# import matplotlib.pyplot as plt  
    
# # store the random numbers in a list  
# nums = []  
# mu = 100
# sigma = 50
    
# for i in range(100000):  
#     temp = random.gauss(1, 0.1)
#     nums.append(temp)  
        
# # plotting a graph  
# plt.hist(nums, bins = 200)  
# plt.show() 
# %%
