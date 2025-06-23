import torch
from torch import is_tensor
import random


class Pad(object):
    def __init__(self, m=None):
        self.m = torch.nn.ReflectionPad2d(4)  # 利用输入边界的反射来填充输入张量。上下左右各填充4。

    def __call__(self, image):
        assert image is not None, "img should not be None"
        assert is_tensor(image), "Tensor expected"
        return self.m(image.unsqueeze(0)).squeeze()


class Crop(object):
    def __init__(self, crop_frac=1.0):
        assert crop_frac <= 1.0, "crop_frac can't be greater than 1.0"
        self.crop_frac = crop_frac

    def __call__(self, img):
        assert img is not None, "img should not be None"
        assert is_tensor(img), "Tensor expected"
        # print(img.size())
        # 处理RGB图像
        # h = img.size(1)
        # w = img.size(2)
        # h2 = int(h * self.crop_frac)
        # w2 = int(w * self.crop_frac)
        # h_range = h - h2
        # w_range = w - w2
        # x, y = random.randint(0, w_range), random.randint(0, h_range)
        #
        # img = img.narrow(1, y, h2).narrow(2, x, w2).clone()  # narrow(dim, start, length)

        # 处理灰度图像
        h = img.size(0)
        w = img.size(1)
        h2 = int(h * self.crop_frac)
        w2 = int(w * self.crop_frac)
        h_range = h - h2
        w_range = w - w2
        x, y = random.randint(0, w_range), random.randint(0, h_range)

        img = img.narrow(0, y, h2).narrow(1, x, w2).clone()  # narrow(dim, start, length)
        return img
