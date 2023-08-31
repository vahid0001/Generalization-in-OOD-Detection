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
bs = 1

for m in models:
    print(m)
    model = timm.create_model(m, pretrained=True, num_classes=0)
    transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    corruptions_augs = os.listdir("ImageNet-30-R/")

    print("train set features")
    train_data = dset.ImageFolder('imagenet30/one_class_train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=False)    

    test_data = dset.ImageFolder('imagenet30/one_class_test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False)

    train_features = []
    test_features = []
    
    train_labels = []
    test_labels = []

    with torch.no_grad():
        for img, label in tqdm(train_loader):
            features = model(img.to(device))
            train_features.append(features.cpu().numpy())
            train_labels.append(label.cpu().numpy())
            del features

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    os.makedirs("ImageNet-30-R_features", exist_ok=True)
    np.save("ImageNet-30-R_features/features_" + m + '_train.npy', train_features)
    np.save("ImageNet-30-R_features/labels" + '_train.npy', train_labels)
    del train_features
    torch.cuda.empty_cache()
    gc.collect()

    print("test set (original) features")

    with torch.no_grad():
        for img, _ in tqdm(test_loader):
            features = model(img.to(device))
            test_features.append(features.cpu().numpy())
            test_labels.append(label.cpu().numpy())
            del features

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    np.save("ImageNet-30-R_features/features_" + m + '_test.npy', test_features)
    np.save("ImageNet-30-R_features/labels" + '_test.npy', test_labels)
    del test_features
    torch.cuda.empty_cache()
    gc.collect()

    for c in corruptions_augs:
        if c == "labels-A.npy" or c == "labels-C.npy":
            continue
        
        print(c)
        corruptions_features = []
        corruptions_labels = []

        data_c = torch.Tensor(np.load("ImageNet-30-R/" + c))
        data = TensorDataset(data_c)
        dataloader_c = DataLoader(data, batch_size=1, shuffle=False)
        to_pil = T.ToPILImage()

        with torch.no_grad():
            for image in tqdm(dataloader_c):
                features = model(transform(to_pil(image[0].numpy().squeeze().astype(np.uint8))).unsqueeze(0).to(device))
                corruptions_features.append(features.cpu().numpy())
                del features
                del image
        
        # torch.cuda.empty_cache()
        # gc.collect()
        
        corruptions_features = np.array(corruptions_features)
        
        np.save("ImageNet-30-R_features/features_" + m + '_' + c, corruptions_features)

        del corruptions_features
        del data_c
        del data
        del dataloader_c
    del model
    # torch.cuda.empty_cache()
    # gc.collect()