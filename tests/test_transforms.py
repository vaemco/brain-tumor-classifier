import torch
from PIL import Image

from brain_tumor.transforms import AddGaussianNoise, build_train_transforms, build_val_transforms


def test_add_gaussian_noise_shape_preserved():
    noise = AddGaussianNoise(0.0, 0.1)
    x = torch.zeros((3, 10, 10))
    out = noise(x)
    assert out.shape == x.shape


def test_train_transform_outputs_tensor():
    tf = build_train_transforms()
    img = Image.new("L", (256, 256), color=128)
    out = tf(img)
    assert out.shape == torch.Size([3, 224, 224])


def test_val_transform_outputs_tensor():
    tf = build_val_transforms()
    img = Image.new("L", (256, 256), color=128)
    out = tf(img)
    assert out.shape == torch.Size([3, 224, 224])
