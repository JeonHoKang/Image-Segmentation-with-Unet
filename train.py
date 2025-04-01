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
# import kagglehub

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] - raw logits
        target: [B, H, W] - class labels
        """
        pred = F.softmax(pred, dim=1)  # convert logits to probabilities
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])  # [B, H, W, C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()     # [B, C, H, W]

        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = torch.sum(pred * target_one_hot, dims)
        union = torch.sum(pred + target_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1. - dice
        return loss.mean()

class Conv2dBlock(nn.Module):
    '''
    conv2d -> GroupNorm -> ReLu
    '''
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels = input_dim, out_channels = out_dim, kernel_size = 3, stride =1 , padding = 3//2),
            nn.GroupNorm(num_groups = 8, num_channels=out_dim),            
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=out_dim)
        )
    def forward(self,x):
        return self.block(x)

class Unet(nn.Module):
    def __init__(self,
                num_classes,
                down_dim = [64, 128, 256, 512]
                ):
        super(Unet, self).__init__()
        self.num_classes = num_classes

        self.down_modules = nn.ModuleList([])
        self.up_modules = nn.ModuleList([])
        all_dims = [3] + list(down_dim)
        start_dim = down_dim[0]
        mid_dim = all_dims[-1]
        

        self.start_dim = start_dim
        self.mid_dim = mid_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out)-1)
            self.down_modules.append(nn.ModuleList([
                Conv2dBlock(dim_in, dim_out),
                Conv2dBlock(dim_out, dim_out),
                nn.MaxPool2d(kernel_size=2, stride =2) if not is_last else nn.Identity()
            ]))

        self.mid_modules = nn.ModuleList([Conv2dBlock(mid_dim, mid_dim*2),
                                Conv2dBlock(mid_dim*2, mid_dim)])            
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out)-1)
            self.up_modules.append(nn.ModuleList([nn.ConvTranspose2d(dim_out*2, dim_in, kernel_size=3, stride = 2,  padding =1 ,output_padding =1),
                                                    Conv2dBlock(dim_in, dim_in),
                                                    Conv2dBlock(dim_in, dim_in)]))
        self.output = nn.Conv2d(start_dim, num_classes, kernel_size=3, stride =1, padding =1)

    def forward(self, x):
        res_h = []
        for idx, (conv_net, conv_net2, downsample) in enumerate(self.down_modules):
            x = conv_net(x)
            x = conv_net2(x)
            res_h.append(x)
            x = downsample(x)
        
        for mid_modules in self.mid_modules:
            x = mid_modules(x)
        
        for idx, (conv_net, conv_net2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, res_h.pop()), dim = 1)
            x = conv_net(x)
            x = conv_net2(x)
            x = upsample(x)
        x = self.output(x)

        return x

class dataset(Dataset):
    def __init__(self, image_dir, label_model, num_classes = 10):
        self.image_dir = image_dir
        self.label_model = label_model
        self.image_fns = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)
        cityscape, label = self.split_image(image)
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
        cityscape = self.transform(cityscape)
        label_class = torch.Tensor(label_class).long()
        return cityscape, label_class
        
    
    def split_image(self, image):
        image = np.array(image)
        cityscape, label = image[:, :256, :], image[:, 256:, :]
        return cityscape, label
    
    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform_ops(image)
    

# def analyze_data(train_dir, train_fns):
#     sample_image_fp = os.path.join(train_dir, train_fns[0])
#     sample_image = Image.open(sample_image_fp).convert("RGB")
#     # plt.imshow(sample_image)
#     # plt.show()
#     print(sample_image_fp)
#     sample_image = np.array(sample_image)
#     print(sample_image.shape)
#     cityscape, label = split_image(sample_image)
#     print(cityscape.min(), cityscape.max(), label.min(), label.max())
#     cityscape, label = Image.fromarray(cityscape), Image.fromarray(label)
#     fig, axes = plt.subplots(1, 2, figsize = (10,5))
#     axes[0].imshow(cityscape)
#     axes[1].imshow(label)
#     plt.show()



# def split_image(image):
#     image = np.array(image)
#     cityscape, label = image[:, :256, :], image[:, 256:, :]
#     return cityscape, label


def train():
    data_dir = os.path.join("cityscapes_data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    train_fns = os.listdir(train_dir)
    val_fns = os.listdir(val_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(device)
    num_classes = 10
    batch_size = 36
    num_items = 1000
    color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)
    data_imported = dataset(train_dir, label_model, num_classes)
    # Analyze data

    cityscape, label_class = data_imported[0]
    print(cityscape.shape, label_class.shape)
    model = Unet(num_classes = num_classes).to(device)
    data_loader = DataLoader(data_imported,
                             batch_size = batch_size)
    
    print(len(data_imported), len(data_loader))

    data_iter = iter(data_loader)
    X, Y = next(data_iter)
    print(X.shape, Y.shape)
    validation_batch_size = 64
    validation_dataset = dataset(val_dir, label_model, num_classes)
    val_data_loader = DataLoader(validation_dataset, batch_size= validation_batch_size)
    X_val,Y_val = next(iter(val_data_loader))
    # Y_pred = model(X)
    # print(Y_pred.shape)
    
    learning_Rate = 1e-3


    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_Rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # Use 'max' if you're monitoring accuracy or IoU
    factor=0.5,       # Reduce LR by a factor (e.g., 0.1 → 10x smaller)
    patience=10       # Wait for 5 epochs of no improvement
    )

    dice_loss = DiceLoss()
    step_losses = []
    epoch_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        model.train()
        for X, Y in tqdm(data_loader, total = len(data_loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        avg_train_loss = epoch_loss / len(data_loader)
        epoch_losses.append(avg_train_loss)
        val_loss = 0
        print(f"\nEpoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        model.eval()
        with torch.no_grad():
            for X_val, Y_val in val_data_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                Y_val_pred = model(X_val)
                loss_val = criterion(Y_val_pred, Y_val)
                val_loss += loss_val.item()
        avg_val_loss = val_loss / len(val_data_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")        
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    
        if (epoch+1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            # Save only the state_dict of the model, including relevant submodules
            torch.save(model.state_dict(),  f'Unet_{epoch+1}_w_decay.pth')
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    model_name = "U-Net.pth"
    torch.save(model.state_dict(), model_name)
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    axes[2].plot(val_losses)
    plt.tight_layout()

    # Save the figure
    fig_path = "loss_curves_with_decay.png"
    plt.savefig(fig_path)
    plt.show()

    # Download latest version
    # path = kagglehub.dataset_download("dansbecker/cityscapes-image-pairs")
    # print(path)

if __name__ == "__main__":
    train()