# Imports
import time
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches

class Plot_Helper():
    def __init__(self):
        self.loss = []
        self.figure_file = 'loss_plot.png'
    
    def record(self,loss):
        if loss != None:
            self.loss.append(loss)
        
    def plot(self):
        fig = plt.figure(1,figsize=(8,8))
        plt.clf()
        gs = fig.add_gridspec(1,1)
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(self.loss,'r')
        ax0.set_xlabel('steps')
        plt.title('loss')
        
        plt.tight_layout()
        plt.pause(0.001)  # pause a bit so that plots are updated
        
        fig.savefig(self.figure_file)
    
    def save_data(self):
        np.save('loss.npy', self.loss)
        
class HonertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 3
learning_rate = 1e-4
batch_size = 16
num_epochs = 400

# Load Data
dataset = HonertDataset(
    csv_file="./dataset/hornet_label.csv",
    root_dir="./dataset/all",
    transform=transforms.ToTensor(),
)

train_set, test_set = torch.utils.data.random_split(dataset, [6720, 8400-6720])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


VGG_types = {
    "VGG11": [64,"M",128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "VGG13": [64,64,"M",128,128,"M",256,256,"M",512,512,"M",512,512,"M"],
    "VGG16": [64,64,"M", 128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M",],
    "VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M",],
}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


# Model
model = VGG_net(in_channels=3, num_classes=num_classes).to(device)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

plot_helper = Plot_Helper()
# Train Network
start_t = time.time()
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        
    plot_helper.record(sum(losses)/len(losses))
    plot_helper.plot()
    msg_tmp = "---epoch:{}".format(epoch)
    msg_tmp += " || loss:{:5.4f}".format(sum(losses)/len(losses))
    msg_tmp += " || time(s):{}".format(int(time.time()-start_t))
    print(msg_tmp)
    torch.save(model.state_dict(),'VGG16.h5')
    # print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)