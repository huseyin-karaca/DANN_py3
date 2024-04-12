import torch
from torch.utils.data import RandomSampler, BatchSampler, DataLoader, random_split, Subset
import numpy as np
from torchvision import datasets, transforms
from huseyin_functions import get_run_name, distribute_apples
import random, os
from data_loader import GetLoader

Ns = 50000
Nt = 50000
batch_size = 128
image_size = 28

Ms = 100
Mt = 25

n_epoch = 300
best_accu_t = 0.0

source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('dataset', source_dataset_name)
target_image_root = os.path.join('dataset', target_dataset_name)

img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])


img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])



Ks = int(np.floor(Ns/batch_size)) # former len_dataloader. how many batches are there in every epoch?
Kt = int(np.floor(Nt/batch_size))

Msx_list = distribute_apples(Ms,Ks)
Mtx_list = distribute_apples(Mt,Kt)


# read source and target datasets for the first time:
dataset_source = datasets.MNIST(
    root='dataset',
    train=True,
    transform=img_transform_source,
    download=True
)

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=os.path.join(target_image_root, 'mnist_m_train_labels.txt'),
    transform=img_transform_target
)

# create subsets of the original source and target datasets. choose random Ns and Nt indices respectively:
dataset_source = Subset(dataset_source, indices=random.sample(range(len(dataset_source)),Ns))
dataset_target = Subset(dataset_target, indices=random.sample(range(len(dataset_target)),Nt))

# split labeled and unlabeled datasets:
dataset_source_labeled, dataset_source_unlabeled = random_split(dataset_source, [Ms, Ns-Ms])
dataset_target_labeled, dataset_target_unlabeled = random_split(dataset_target, [Mt, Nt-Mt])

# create labeled and unlabeled dataloaders seperately:
# (in order to prevent using more labeled total data than specified Ms and Mt values)

# labeled source dataloader's batch size is max(Msx_list) as Msx_list defines the 
# distribution of number of labeled samples per batch. when creating merged source batch, 
# we will replace the last max(Msx_list) image of the unlabeled source batch with the 
# images coming from the labeled dataloader. 
# target dataloaders will also undergo the same process.

# source dataloaders
dataloader_source_labeled = DataLoader(
    dataset=dataset_source_labeled,
    batch_size=max(Msx_list),
    shuffle=True,
    num_workers=8,
    drop_last = True)
    
dataloader_source_unlabeled = DataLoader(
    dataset=dataset_source_unlabeled,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last = True)

# target dataloaders
dataloader_target_labeled = DataLoader(
    dataset=dataset_target_labeled,
    batch_size=max(Mtx_list),
    shuffle=True,
    num_workers=8,
    drop_last = True)
    
dataloader_target_unlabeled = DataLoader(
    dataset=dataset_target_unlabeled,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last = True)

# training

for epoch in range(n_epoch):

    source_labeled_iter = iter(dataloader_source_labeled)
    source_unlabeled_iter = iter(dataloader_source_unlabeled)

    for i in range(min(Ks,Kt)): # Ks, Kt: dataloader lengths if a non-iterative dataloader would have been used
        Msx = Msx_list[i] # required number of labeled source data for per batch to ensure total of Ms 
        Mtx = Mtx_list[i] 

        source_labeled_data = source_labeled_iter.next()
        source_unlabeled_data = source_unlabeled_iter.next()
