import torch
import numpy as np
import math
import random

import torchvision
import shutil
import cv2
from torchvision import transforms
from torchvision.transforms import v2 as T
from PIL import Image, ImageEnhance, ImageOps
def HFlip(sat, grd = None):
    #使用 [2] 时，它是在高度维度上翻转张量，这将导致图像上下颠倒。
    print("hflip, sat.shape:",sat.shape)
    if sat.dim() == 3:
        h_dim = [1]
    elif sat.dim() == 4:
        h_dim = [2]
    if sat is not None and grd is not None:
        sat = torch.flip(sat, h_dim)
        grd = torch.flip(grd, h_dim)
        return sat, grd

    elif sat is not None:
        sat = torch.flip(sat, h_dim)
        return sat

    elif grd is not None:
        grd = torch.flip(grd, h_dim)
        return  grd

def augmentation(aug):
    final = []
    if 'randomresizedcrop' in aug:
        final.append(t.RandomResizedCrop(size=(224, 224)))
    elif 'randomhorizontflip' in aug:
        final.append(T.RandomHorizontalFlip(p=0.5))
    elif 'colorjitter' in aug:
        final.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
    elif 'randomphotometricdistort' in aug:
        final.append(T.RandomPhotometricDistort)
    elif 'grayscale' in aug:
        final.append(T.Grayscale())
    elif 'randomgrayscale' in aug:
        final.append(T.RandomGrayscale())
    elif 'gaussianblur' in aug:
        final.append(T.GaussianBlur(kernel_size=1))
    elif 'gaussiannoise' in aug:
        final.append(T.GaussianNoise())
    trans = T.Compose(final)
    return trans

def identity_func(img):
    return img


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256

    def tune_channel(ch):
        hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
        non_zero_hist = hist[hist != 0].reshape(-1)
        step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
        if step == 0: return ch
        n = np.empty_like(hist)
        n[0] = step // 2
        n[1:] = hist[:-1]
        table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    like PIL, rotate by degree, not radians
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out

#曝光过度Solarization
def solarize_func(img, thresh=128):
    '''
        same output as PIL.ImageOps.posterize
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    ## implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
            np.float32([
                [0.886, -0.114, -0.114],
                [-0.587, 0.413, -0.587],
                [-0.299, -0.299, 0.701]]) * factor
            + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out

#水平错切（Horizontal Shear）
def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def posterize_func(img, bits):
    '''
        same output as PIL.ImageOps.posterize
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


### level to args
def enhance_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        return ((level / MAX_LEVEL) * 1.8 + 0.1,)
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 0.3
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * float(translate_const)
        if np.random.random() > 0.5: level = -level
        return (level, replace_value)

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * cutout_const)
        return (level, replace_value)

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 256)
        return (level, )
    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(MAX_LEVEL):
    def level_to_args(level):
        level = int((level / MAX_LEVEL) * 4)
        return (level, )
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):
    def level_to_args(level):
        level = (level / MAX_LEVEL) * 30
        if np.random.random() < 0.5:
            level = -level
        return (level, replace_value)

    return level_to_args

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)

arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    # 'ShearX': shear_level_to_args(MAX_LEVEL, replace_value), # 水平或垂直错切。让图像沿某个轴倾斜，模拟从不同角度观察物体。
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    # 'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}
func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    # 'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    # 'ShearY': shear_y_func,
}

class RandomAugment(object):

    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        self.N = N
        self.M = M
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]

    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img

import numpy as np


# ... (arg_dict, func_dict, RandomAugment 类定义保持不变) ...

class RangeRandomAugment(object):

    def __init__(self, N=2, M_range=(0, 10), isPIL=False, augs=[]):
        self.N = N
        self.M_range = M_range  # M 现在是一个元组，定义了强度的范围
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.geo_augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]


    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)

        # 关键修正：在每次调用时，从 M_range 中随机选择一个整数作为强度 M
        M = np.random.randint(self.M_range[0], self.M_range[1] + 1)

        ops = self.get_random_ops()
        for name, prob, _ in ops:  # 注意，这里不再使用固定的 M
            if np.random.random() > prob:
                continue
            args = arg_dict[name](M)  # 将随机选择的 M 传入
            img = func_dict[name](img, *args)
        return img

def geotext_train_transform():
    train_transform = T.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),


    ])
    return train_transform

def random_transform():
    train_transform = T.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity','Equalize', 'Brightness', 'Sharpness']),
    ])
    return train_transform


def puregeo_train_transform(image_size=224):
    transform = T.Compose([
        T.RandomAffine(degrees=(-15, 15),translate=(0.05, 0.1),scale=(0.9, 1.1),\
                       shear=(-10, 10),interpolation=T.InterpolationMode.BILINEAR),
        RandomPerspectiveWithRange(0.05, 0.1, p=0.5),
        # 5.ElasticTransform
        T.RandomApply( [T.ElasticTransform(alpha=(40.0, 60.0), \
                                           sigma=(8.0, 12.0))],p=0.3)
    ])
    return transform

def geometry_train_transform(image_size=224):
    # 1.second delete
    color_transforms = T.RandomChoice([
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomEqualize(p=0.2),
        RandomSharpness(min_factor=0.8, max_factor=1.2, p=0.2),
    ])
    # 2.third delete
    transform = T.Compose([
        # T.RandomApply control the probability of RandomChoice
        T.RandomApply([color_transforms], p=0.2),
        T.RandomAffine(degrees=(-15, 15),translate=(0.05, 0.1),scale=(0.9, 1.1),\
                       shear=(-10, 10),interpolation=T.InterpolationMode.BILINEAR),
        RandomPerspectiveWithRange(0.05, 0.1, p=0.5),
        # 3. first delete

        # 5.ElasticTransform
        T.RandomApply( [T.ElasticTransform(alpha=(40.0, 60.0), \
                                           sigma=(8.0, 12.0))],p=0.3)
    ])
    return transform

def geometry_without_Elastic_train_transform(image_size=224):
    # 1.second delete
    color_transforms = T.RandomChoice([
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomEqualize(p=0.2),
        RandomSharpness(min_factor=0.8, max_factor=1.2, p=0.2),
    ])
    # 2.third delete
    transform = T.Compose([
        # T.RandomApply control the probability of RandomChoice
        T.RandomApply([color_transforms], p=0.2),
        T.RandomAffine(degrees=(-15, 15),translate=(0.05, 0.1),scale=(0.9, 1.1),\
                       shear=(-10, 10),interpolation=T.InterpolationMode.BILINEAR),
        RandomPerspectiveWithRange(0.05, 0.1, p=0.5),
    ])
    return transform

def geometry_without_Color_train_transform(image_size=224):
    # 2.third delete
    transform = T.Compose([
        # T.RandomApply control the probability of RandomChoice
        T.RandomAffine(degrees=(-15, 15),translate=(0.05, 0.1),scale=(0.9, 1.1),\
                       shear=(-10, 10),interpolation=T.InterpolationMode.BILINEAR),
        RandomPerspectiveWithRange(0.05, 0.1, p=0.5),
        # 3. first delete
        # 5.ElasticTransform
        T.RandomApply( [T.ElasticTransform(alpha=(40.0, 60.0), \
                                           sigma=(8.0, 12.0))],p=0.3)
    ])
    return transform

def geometry_without_ColorAndElastic_train_transform(image_size=224):

    # 2.third delete
    transform = T.Compose([
        # T.RandomApply control the probability of RandomChoice
        T.RandomAffine(degrees=(-15, 15),translate=(0.05, 0.1),scale=(0.9, 1.1),\
                       shear=(-10, 10),interpolation=T.InterpolationMode.BILINEAR),
        RandomPerspectiveWithRange(0.05, 0.1, p=0.5),

    ])
    return transform

class RandomSharpness:
    def __init__(self, min_factor=0.5, max_factor=2.0, p=0.5):
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(self.min_factor, self.max_factor)
            return torchvision.transforms.functional.adjust_sharpness(img, factor)
        return img

class RandomPerspectiveWithRange:
    def __init__(self, min_scale=0.05, max_scale=0.1, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        return T.RandomPerspective(distortion_scale=scale, p=self.p)(img)

# def geomatry_train_transform(image_size=224):
#     transform = A.Compose([
#         # 1. 随机缩放 + 裁剪 + 翻转
#         A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
#         A.HorizontalFlip(p=0.5),
#
#         # 2. 颜色增强（随机选一个）
#         A.OneOf([
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
#             A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0),
#             A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
#             A.Equalize(p=1.0),
#             A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3), p=1.0),
#         ], p=0.7),  # 每次 70% 概率选一种
#
#         # 3. 几何变换
#         A.Affine(
#             scale=(0.9, 1.1),
#             translate_percent=(0.05, 0.1),
#             rotate=(-15, 15),
#             shear=(-10, 10),
#             p=0.7
#         ),
#
#         # 4. 透视变换
#         A.Perspective(scale=(0.05, 0.1), p=0.5),
#
#         # 5. 弹性扭曲 (warping)
#         A.ElasticTransform(alpha=50, sigma=10, alpha_affine=20, p=0.3),
#
#     ])
#     return transform


def Rotate(sat, orientation, grd = None):
    if sat is not None and grd is not None:
        height, width = grd.shape[1], grd.shape[2]
        if orientation == 'left':
            sat_rotate = torch.rot90(sat, -1, [1, 2])
            left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
            right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        elif orientation == 'right':
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
            right_grd = grd[:, :, int(math.floor(width * 0.25)):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        elif orientation == 'back':
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1, 2])
            left_grd = grd[:, :, 0:int(width * 0.5)]
            right_grd = grd[:, :, int(width * 0.5):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return sat_rotate, grd_rotate

    elif sat is not None:
        if orientation == 'left':
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        elif orientation == 'right':
            sat_rotate = torch.rot90(sat, 1, [1, 2])
        elif orientation == 'back':
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1, 2])
        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return sat_rotate

    elif grd is not None:
        height, width = grd.shape[1], grd.shape[2]
        if orientation == 'left':
            left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
            right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        elif orientation == 'right':
            left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
            right_grd = grd[:, :, int(math.floor(width * 0.25)):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        elif orientation == 'back':
            left_grd = grd[:, :, 0:int(width * 0.5)]
            right_grd = grd[:, :, int(width * 0.5):]
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)
        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return grd_rotate

def Rotate_tensor(sat, orientation,grd = None):
    if sat is not None and grd is not None:
        height, width = grd.shape[1], grd.shape[2]
        if orientation == 'left':
            split_width = int(math.ceil(width * 0.75))
            sat_rotate = torch.rot90(sat, -1, [1, 2])
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        elif orientation == 'right':
            split_width = int(math.floor(width * 0.25))
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        elif orientation == 'back':
            split_width = int(width * 0.5)
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1, 2])
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return sat_rotate, grd_rotate
    elif sat is not None:
        if orientation == 'left':
            sat_rotate = torch.rot90(sat, -1, [1, 2])

        elif orientation == 'right':
            sat_rotate = torch.rot90(sat, 1, [1, 2])

        elif orientation == 'back':
            sat_rotate = torch.rot90(sat, 1, [1, 2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1, 2])

        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return sat_rotate

    elif grd is not None:
        height, width = grd.shape[1], grd.shape[2]
        if orientation == 'left':
            split_width = int(math.ceil(width * 0.75))
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        elif orientation == 'right':
            split_width = int(math.floor(width * 0.25))
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        elif orientation == 'back':
            split_width = int(width * 0.5)
            left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
            grd_rotate = torch.cat([right_grd, left_grd], dim=2)

        else:
            raise RuntimeError(f"Orientation {orientation} is not implemented")

        return grd_rotate

def Reverse_Rotate_Flip(sat, perturb,grd = None):
    # Reverse process
    if sat is not None and grd is not None:
        assert sat.shape[0] == grd.shape[0]
        assert sat.shape[0] == len(perturb)

        sat = sat.permute(0,3,1,2)
        grd = grd.permute(0,3,1,2)

        reversed_sat_desc = torch.zeros_like(sat)
        reversed_grd_desc = torch.zeros_like(grd)

        for i in range(len(perturb)):
            reverse_perturb = [None, None]
            reverse_perturb[0] = perturb[i][0]

            if perturb[i][1] == "left":
                reverse_perturb[1] = "right"
            elif perturb[i][1] == "right":
                reverse_perturb[1] = "left"
            else:
                reverse_perturb[1] = perturb[i][1]

            # print(reverse_perturb)
            # reverse process first rotate then flip
            if reverse_perturb[1] != "none":
                rotated_sati, rotated_grdi = Rotate_tensor(sat[i], grd[i], reverse_perturb[1])
            else:
                rotated_sati = sat[i]
                rotated_grdi = grd[i]

            if reverse_perturb[0] == 1:
                reversed_sat_desc[i], reversed_grd_desc[i] = HFlip(rotated_sati, rotated_grdi)
            else:
                reversed_sat_desc[i] = rotated_sati
                reversed_grd_desc[i] = rotated_grdi

        reversed_sat_desc = reversed_sat_desc.permute(0,2,3,1)
        reversed_grd_desc = reversed_grd_desc.permute(0,2,3,1)

        return reversed_sat_desc, reversed_grd_desc
    elif sat is not None:
        assert sat.shape[0] == len(perturb)

        sat = sat.permute(0, 3, 1, 2)

        reversed_sat_desc = torch.zeros_like(sat)

        for i in range(len(perturb)):
            reverse_perturb = [None, None]
            reverse_perturb[0] = perturb[i][0]

            if perturb[i][1] == "left":
                reverse_perturb[1] = "right"
            elif perturb[i][1] == "right":
                reverse_perturb[1] = "left"
            else:
                reverse_perturb[1] = perturb[i][1]

            if reverse_perturb[1] != "none":
                rotated_sati = Rotate_tensor(sat[i], reverse_perturb[1])
            else:
                rotated_sati = sat[i]

            if reverse_perturb[0] == 1:
                reversed_sat_desc[i] = HFlip(rotated_sati)
            else:
                reversed_sat_desc[i] = rotated_sati

        reversed_sat_desc = reversed_sat_desc.permute(0, 2, 3, 1)

        return reversed_sat_desc

    elif grd is not None:
        grd = grd.permute(0, 3, 1, 2)
        reversed_grd_desc = torch.zeros_like(grd)
        for i in range(len(perturb)):
            reverse_perturb = [None, None]
            reverse_perturb[0] = perturb[i][0]

            if perturb[i][1] == "left":
                reverse_perturb[1] = "right"
            elif perturb[i][1] == "right":
                reverse_perturb[1] = "left"
            else:
                reverse_perturb[1] = perturb[i][1]

            if reverse_perturb[1] != "none":
                rotated_grdi = Rotate_tensor(grd[i], reverse_perturb[1])
            else:
                rotated_grdi = grd[i]

            if reverse_perturb[0] == 1:
                reversed_grd_desc[i] = HFlip(rotated_grdi)
            else:
                reversed_grd_desc[i] = rotated_grdi

        reversed_grd_desc = reversed_grd_desc.permute(0, 2, 3, 1)

        return reversed_grd_desc


def load_and_preprocess_image(path):
    """
    加载图片并将其转换为 PyTorch 张量。
    """
    # 1. 加载图片
    img = Image.open(path).convert('RGB')  # 确保是RGB格式
    # 2. 转换为 NumPy 数组
    np_img = np.array(img).astype(np.float32)
    # 3. 转换为 PyTorch 张量，并调整到 (height, width, channels)
    tensor_img = torch.from_numpy(np_img) / 255.0
    # 4. 调整维度以匹配代码
    # 代码期望 (batch_size, height, width, channels)
    return tensor_img.unsqueeze(0)


def save_tensor_as_image(tensor, path):
    """
    将 PyTorch 张量保存为本地图片文件。
    :param tensor: 形状为 (1, H, W, C) 的 PyTorch 张量。
    :param path: 保存图片的本地路径，例如 "output.jpg"。
    """
    # 1. 移除批次维度
    # 从 (1, H, W, C) 变为 (H, W, C)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    # 2. 反归一化并转换为整型
    # 值范围从 [0, 1] 恢复到 [0, 255]
    numpy_array = (tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)
    # 3. 转换为 PIL 图像并保存
    img = Image.fromarray(numpy_array)
    img.save(path)
    print(f"图片已成功保存到: {path}")


def save_tensor_as_image_(tensor, path):
    """
    将 PyTorch 张量保存为本地图片文件。
    :param tensor: 形状为 (1, H, W, C) 的 PyTorch 张量。
    :param path: 保存图片的本地路径，例如 "output.jpg"。
    """
    # 1. 移除批次维度
    # 从 (1, H, W, C) 变为 (H, W, C)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    # 2. 反归一化并转换为整型
    # 值范围从 [0, 1] 恢复到 [0, 255]
    numpy_array = (tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)
    # 3. 转换为 PIL 图像并保存
    img = Image.fromarray(numpy_array)
    img.save(path)
    print(f"图片已成功保存到: {path}")


if __name__ == "__main__":
    sat_image_path = "/gpfs2/scratch/ychen57/Datasets/CVGText/images/NewYork-satellite/40.70682453,-73.99794426_2023-05_kmLvJZpQCpMcd0aqM93H-A_d44_z3.png"
    save_path = "/gpfs2/scratch/ychen57/Datasets/aug/"
    destination_path = save_path +"ori_img.jpg"
    shutil.copyfile(sat_image_path, destination_path)
    print(f"原始图片已成功保存到: {destination_path}")
    try:
        sat = load_and_preprocess_image(sat_image_path)
        print("sat.shape:", sat.shape)
        # batch_size = 1
        sat = sat.repeat(6, 1, 1, 1)
        # grd = grd.repeat(batch_size, 1, 1, 1)

    except FileNotFoundError:
        print("图片文件未找到，请检查路径。")
        # 如果文件不存在，可以回退到使用随机张量进行测试
        sat = torch.rand(32, 8, 42, 8)

    # Copy to generate a new descriptor
    mu_sat = sat.clone().detach()
    print("mu_sat.shape:", mu_sat.shape)
    # generate new descriptor by LS
    mu_sat = mu_sat.permute(0, 3, 1, 2)
    print("after permute, mu_sat.shape:", mu_sat.shape)
    first_sat = torch.zeros_like(mu_sat)

    # generate # of batch size LS operations
    perturb = [[1, "none"], [1, "left"], [1, "back"], [0, "right"], [0, "left"], [0, "back"]]
    # perturb = []
    # for i in range(32):
    #     hflip = random.randint(0, 1)
    #     orientation = random.choice(["left", "right", "back", "none"])
    #
    #     while hflip == 0 and orientation == "none":
    #         hflip = random.randint(0, 1)
    #         orientation = random.choice(["left", "right", "back", "none"])
    #
    #     perturb.append([hflip, orientation])

    print(perturb)

    # perform LS to generate new layout
    for i in range(len(perturb)):
        orig_sat = mu_sat
        if perturb[i][0] == 1:
            orig_sat = HFlip(orig_sat)
        if perturb[i][1] != "none":
            first_sat[i] = Rotate(orig_sat[i], perturb[i][1])
        else:
            first_sat[i] = orig_sat[i]
        save_name = str(perturb[i])
        # (H, W, C) 或 (C, H, W)。
        print("first_sat[i].shape:", first_sat[i].shape)
        save_tensor_as_image(first_sat[i], save_path+save_name+".jpg")

    first_sat = first_sat.permute(0, 2, 3, 1)

    print("=====before:")
    # print(grd[0, :, :, :])
    # print(first_grd[0, :, :, :])
    # print(sat[0, :, :, :])
    # print(first_sat[0, :, :, :])
    print(torch.equal(sat, first_sat))

    # Reverse to original layout
    second_sat = Reverse_Rotate_Flip(first_sat, perturb)
    for i in range(len(perturb)):
        save_name = str(perturb[i])+"_reverse.jpg"
        save_tensor_as_image_(second_sat[i], save_path+save_name)
    print("=====after:")
    # print(grd[0, :, :, :])
    # print(second_grd[0, :, :, :])
    # print(sat[0, :, :, :])
    # print(second_sat[0, :, :, :])
    print(torch.equal(sat, second_sat))
    for i in range(sat.shape[0]):
        print(f"=============={i}==============")


        print(torch.allclose(sat[i], second_sat[i]))
        if not torch.equal(sat[i], second_sat[i]):
            print(f"difference in {i}:",sat[i]-second_sat[i])
        print("================================")