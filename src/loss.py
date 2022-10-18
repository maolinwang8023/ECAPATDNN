"""loss function"""
import math
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np


class AdditiveAngularMargin(nn.Cell):
    """AAM lossfunction"""
    def __init__(self, margin=0.0, scale=1.0):
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.sqrt = ops.Sqrt()
        self.pow = ops.Pow()

    def construct(self, outputs, targets):
        cosine = outputs
        sine = self.sqrt(1.0 - self.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = np.where(cosine > self.th, phi, cosine - self.mm)
        output = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * output
