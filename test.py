import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from train import Unet, dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)    
data_dir = os.path.join("cityscapes_data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_path = "/home/jeonkang/jeon_ws/Image_Segmentation/Unet_10_w_decay.pth"

num_classes = 10

model_ = Unet(num_classes = num_classes).to(device)
num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)

model_.load_state_dict(torch.load(model_path))
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)
test_batch = 3
city_dataset = dataset(test_dir, label_model=label_model)
data_loader = DataLoader(city_dataset, batch_size= test_batch, shuffle=True)

X,Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred = torch.argmax(Y_pred, dim=1)
print(Y_pred.shape)
inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])
fig, axes = plt.subplots(test_batch, 3, figsize=(3*5, test_batch*5))

for i in range(test_batch):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    # plt.imshow(landscape)
    
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")    
plt.show()