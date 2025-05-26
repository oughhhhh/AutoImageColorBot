import argparse
import os
import base64
from io import BytesIO
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2gray
import PIL.Image


def to_rgb(grayscale_input, ab_input):
    # Show/save rgb image from grayscale and ab channels
    plt.clf() # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    plt.imsave(arr=color_image, fname='temp.jpg')


# def colorize_image(image_bytes: bytes) -> bytes:
#     image = PIL.Image.open(BytesIO(image_bytes)).convert("RGB")

#     colorized_image = image

#     output = BytesIO()
#     colorized_image.save(output, format="JPEG")
#     return output.getvalue()


def colorize(image_bytes : bytes) -> bytes:
    image = PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
    input_size = image.size
    image.save("temp.jpg")

    print('Beginning Inference')
    model = torch.load("models/saved_model.pth", weights_only=False)
    input_gray = cv2.imread("temp.jpg")
    input_gray = cv2.resize(input_gray, (256,256))
    input_gray = rgb2gray(input_gray)
    input_gray = torch.from_numpy(input_gray).unsqueeze(0).float()
    input_gray = torch.unsqueeze(input_gray, dim=0).cuda()
    model.eval()
    output_ab = model(input_gray)
    to_rgb(input_gray[0].cpu(), output_ab[0].detach().cpu())

    output = BytesIO()
    colorized_image = PIL.Image.open("temp.jpg").resize(input_size)
    colorized_image.save(output, format="JPEG")
    return output.getvalue()
