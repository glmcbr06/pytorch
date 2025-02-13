import torch
from torch.utils.data import Dataset
from torchvision import datasets as vis_datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from torchtext import datasets as txt_datasets


# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

train_data = vis_datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)


test_data = vis_datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# create a validate sample from training dataset
indices = list(range(len(train_data)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(train_data)))
train_sample = SubsetRandomSampler(indices[split:])
validate_sample = SubsetRandomSampler(indices[:split])

# data loader
trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sample, batch_size=64)
validloader = torch.utils.data.DataLoader(train_data, sampler=validate_sample, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# create data iterator
dataiter = iter(trainloader)
print(dataiter)
images, labels = next(dataiter)


f = plt.figure(figsize=(15, 5))
for i in np.arange(20):
    ax = f.add_subplot(4, int(20/4), i + 1)
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    f.tight_layout()
plt.show()



# def main(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     main('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
