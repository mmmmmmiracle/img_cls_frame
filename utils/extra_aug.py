import albumentations as albu 
import cv2 
from PIL import Image
import numpy as np

BORDER_MODE = [
   cv2.BORDER_REPLICATE, 
   cv2.BORDER_CONSTANT, 
   cv2.BORDER_REFLECT, 
   cv2.BORDER_WRAP, 
   cv2.BORDER_REFLECT_101
]

def aug(img, aug_func):
    return aug_func(**{'image':img})['image']

class GridDistortion():
    def __init__(self, num_steps=5, distort_limit=0.3, p=0.5, border_mode = 1):
        self.num_steps = num_steps
        self.distort_limit = distort_limit 
        self.p = p
        self.border_mode = BORDER_MODE[border_mode]

    def __call__(self, img):    
        img = aug(np.array(img), albu.GridDistortion(
            num_steps=self.num_steps, 
            distort_limit = self.distort_limit, 
            p = self.p, 
            border_mode = self.border_mode
            ))
        img = Image.fromarray(img)
        return img

class OpticalDistortion():
    def __init__(self, distort_limit=0.05, shift_limit=0.05, border_mode=1, p=0.5):
        self.distort_limit = distort_limit 
        self.shift_limit = shift_limit
        self.border_mode = BORDER_MODE[border_mode]
        self.p = p

    def __call__(self, img):
        # img = aug(np.array(img), albu.OpticalDistortion(border_mode = cv2.BORDER_CONSTANT))
        img = aug(np.array(img), albu.OpticalDistortion(
                distort_limit = slef.distort_limit,
                shift_limit = self.shift_limit, 
                border_mode = self.border_mode, 
                p = self.p
            ))
        img = Image.fromarray(img)
        return img

class ElasticTransform():
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, border_mode=1, p=0.5):
        self.alpha = alpha
        self.sigma = sigma 
        self.alpha_affine = alpha_affine
        self.border_mode = BORDER_MODE[border_mode]
        self.p = p

    def __call__(self, img):
        img = aug(np.array(img), albu.ElasticTransform(
                alpha=self.alpha, sigma=self.sigma, 
                alpha_affine=self.alpha_affine,
                border_mode=self.border_mode,
                p=self.p
            ))
        img = Image.fromarray(img)
        return img

class RandomShadow():
    def __call__(self, img):
        img = aug(np.array(img), albu.RandomShadow())
        img = Image.fromarray(img)
        return img
    
class GaussNoise():
    def __call__(self, img):
        img = aug(np.array(img), albu.GaussNoise())
        img = Image.fromarray(img)
        return img

class RandomBrightness():
    def __call__(self, img):
        img = aug(np.array(img), albu.RandomBrightness())
        img = Image.fromarray(img)
        return img
    
class RandomGamma():
    def __call__(self, img):
        img = aug(np.array(img), albu.RandomGamma())
        img = Image.fromarray(img)
        return img
    
class RandomFog():
    def __call__(self, img):
        img = aug(np.array(img), albu.RandomFog())
        img = Image.fromarray(img)
        return img

class RandomRain():
    def __call__(self, img):
        img = aug(np.array(img), albu.RandomBrightness())
        img = Image.fromarray(img)
        return img

class IAAAffine():
    def __init__(self, scale=1.0, rotate=0.0, shear=0.0, mode='constant', p=0.5):
        self.scale = scale
        self.rotate = rotate
        self.shear = shear
        self.mode = mode
        self.p = p

    def __call__(self, img):
        img = aug(np.array(img), albu.IAAAffine(
                scale=self.scale, rotate=self.rotate, shear=self.shear, mode=self.mode, p=self.p
            ))
        img = Image.fromarray(img)
        return img

class ShiftScaleRotate():
    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, border_mode=1, p=0.5):
        self.shift_limit = shift_limit
        self.scale_limit =scale_limit
        self.rotate_limit = rotate_limit
        self.border_mode = BORDER_MODE[border_mode] 
        self. p = p

    def __call__(self, img):
        img = aug(np.array(img), albu.ShiftScaleRotate(
                shift_limit=self.shift_limit, scale_limit=self.scale_limit, 
                rotate_limit=self.rotate_limit, border_mode=self.border_mode,
                p=self.p
            ))
        img = Image.fromarray(img)
        return img