import torch
import numpy as np
from skimage.color import rgb2lab, rgb2gray
import datasets
import torchvision.transforms as T


class ColorizeData():
    def __init__(self, ds : datasets.arrow_dataset.Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)


    def __getitem__(self, index):
        input = self.ds[index]["image"]
        if input.shape[0] != 3: input = torch.zeros((3, 244, 244))

        transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
        input = transform(input)

        input = np.asarray(input)
        input = input.transpose((1, 2, 0))

        img_lab = rgb2lab(input)
        # Normalize the LAB values so that they are in the range of 0 to 1
        img_lab = (img_lab + 128) / 255
        # Get the "ab" channels from the LAB image
        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        input = rgb2gray(input)
        input = torch.from_numpy(input).unsqueeze(0).float()

        # Return the grayscale image and the "ab" channels as a tuple
        return input, img_ab
