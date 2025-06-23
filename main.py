import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from data_loader import getTargetDataSet
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import data_loader
from lib.transforms import Pad, Crop
from models.ConvolutionalAutoencoder import ConvAutoencoder


def main():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('./trained_cae/resnet_cifar10_FGSM-0.1.pth'))
    model.eval()
    print(model)
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size(), parameters)


    data = torch.tensor(torch.load('./adv_output/resnet_cifar10/train/Train_clean_data_resnet_cifar10_FGSM-0.1.pth'))
    # print(data[10])
    t = transforms.ToPILImage('RGB')
    img = data[520]

    # trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])
    # img = plt.imread('./0_image.png')
    # img = trans(img)
    re_img = model(img)
    # # print(img)
    plt.title('original image')
    img = t(img)
    plt.imshow(img)
    plt.show()


    plt.title('processed image')
    re_img = t(re_img)
    plt.imshow(re_img)
    plt.show()


if __name__ == '__main__':
    main()
