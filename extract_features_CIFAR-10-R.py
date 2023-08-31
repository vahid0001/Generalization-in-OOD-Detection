""""

pip install git+https://github.com/rwightman/pytorch-image-models

"""

import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms as T
import timm
import os
import numpy as np
from tqdm import tqdm
import torch
import gc

models = ['vit_base_patch16_224', 'vit_large_patch16_224', 'resnet50', 'resnet152']
bs = 50

for m in models:
    print(m)
    model = timm.create_model(m, pretrained=True, num_classes=0)
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    corruptions_augs = os.listdir("CIFAR-10-R/")

    print("train set features")
    train_data = dset.CIFAR10("./", download=True, train=True, transform=transform)
    test_data = dset.CIFAR10("./", download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)

    train_features = []
    test_features = []

    with torch.no_grad():
        for img, _ in tqdm(train_dataloader):
            features = model(img.to(device))
            train_features.append(features.cpu().numpy())
            del features

    train_features = np.array(train_features)
    os.makedirs("CIFAR-10-R_features", exist_ok=True)
    np.save("CIFAR-10-R_features/features_" + m + '_train.npy', train_features)
    del train_features
    torch.cuda.empty_cache()
    gc.collect()

    print("test set (original) features")

    with torch.no_grad():
        for img, _ in tqdm(test_dataloader):
            features = model(img.to(device))
            test_features.append(features.cpu().numpy())
            del features

    test_features = np.array(test_features)
    np.save("CIFAR-10-R_features/features_" + m + '_test.npy', test_features)
    del test_features
    torch.cuda.empty_cache()
    gc.collect()

    for c in corruptions_augs:
        if c == "labels-A.npy" or c == "labels-C.npy":
            continue
        print(c)
        corruptions_features = []
        corruptions_labels = []

        data_c = np.load("CIFAR-10-R/" + c)
        data_c = torch.Tensor(data_c)
        data = TensorDataset(data_c)
        dataloader_c = DataLoader(data, batch_size=1, shuffle=False)
        to_pil = T.ToPILImage()

        with torch.no_grad():
            for image in tqdm(dataloader_c):
                features = model(transform(to_pil(image[0].numpy().squeeze().astype(np.uint8))).unsqueeze(0).to(device))
                corruptions_features.append(features.cpu().numpy())
                del features
        
        torch.cuda.empty_cache()
        gc.collect()

        corruptions_features = np.array(corruptions_features)

        np.save("CIFAR-10-R_features/features_" + m + '_' + c, corruptions_features)

    del corruptions_features
    del model
    torch.cuda.empty_cache()
    gc.collect()