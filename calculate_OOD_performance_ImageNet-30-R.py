"""
!pip install faiss-gpu

"""

import faiss
from sklearn.metrics import roc_auc_score
import numpy as np

output_dim = [768, 1024, 2048, 2048]
models = ['vit_base_patch16_224', 'vit_large_patch16_224', 'resnet50', 'resnet152']

def read_features(aug, corrup, model, o_dim):
    train_features = np.load("ImageNet-30-R_features/features_" + model + "_train.npy").reshape(-1, o_dim)
    train_labels = np.load("ImageNet-30-R-labels_train.npy").squeeze()

    test_features = np.load("ImageNet-30-R_features/features_" + model + "_test.npy").reshape(-1, o_dim)
    test_labels = np.load("ImageNet-30-R-labels_test.npy").squeeze()

    near_in_features = []
    near_in_labels = []
    if aug == True:
        near_in = ["random_crop.npy", "rot90.npy", "rot270.npy", "random_crop.npy", "color_jitter.npy"]
        for c in near_in:
            near_in_features.append(np.load("ImageNet-30-R_features/features_" + model + '_' + c).reshape(3000, o_dim))
            near_in_labels.append(np.load("ImageNet-30-R-labels_test.npy").reshape(3000,))
    if corrup == True:
        near_in = ["brightness.npy", "contrast.npy", "defocus_blur.npy", "elastic_transform.npy",
                   "fog.npy", "frost.npy", "gaussian_blur.npy", "gaussian_noise.npy",
                   "glass_blur.npy", "impulse_noise.npy", "jpeg_compression.npy",
                    "motion_blur.npy", "pixelate.npy", "saturate.npy", "shot_noise.npy",
                    "snow.npy", "spatter.npy", "speckle_noise.npy", "zoom_blur.npy"]
        for c in near_in:
            temp = np.load("ImageNet-30-R_features/features_" + model + '_' + c).reshape(15000, o_dim)
            for i in range(0, 15000, 3000):
                near_in_features.append(temp[i:i+3000])
                near_in_labels.append(np.load("ImageNet-30-R-labels_test.npy").reshape(3000,))



    near_in_features = np.array(near_in_features)
    near_in_labels = np.array(near_in_labels)

    return train_features, train_labels, test_features, test_labels, near_in_features, near_in_labels


for i in range(4):

    train_features, train_labels, test_features, test_labels, \
    near_in_features, near_in_labels = read_features(aug=True, corrup=False, model=models[i], o_dim=output_dim[i])
    AUC = []
    print("Model: " + models[i])
    print("Calculating AUC for Augmentations (A):")

    for normal_class in range(30):
        y = train_labels.copy()
        X = train_features.copy()[y == normal_class].astype("float32")

        y_test = test_labels.copy()
        X_test = test_features.copy().astype("float32")

        inl = X_test[y_test == normal_class]
        inl2 = near_in_features[near_in_labels == normal_class]
        out = X_test[y_test != normal_class]
        
        ############ adaptation
        X_norms = np.linalg.norm(X, axis=1)
        X = X / X_norms[:, np.newaxis]

        data = np.concatenate((inl, inl2, out)).astype("float32")
        labels = np.concatenate((np.zeros((inl.shape[0])), np.zeros((inl2.shape[0])), np.ones((out.shape[0])))).astype(int)

        data_norms = np.linalg.norm(data, axis=1)
        data = data / data_norms[:, np.newaxis]

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        D, _ = index.search(data, 1)
        ano_score = np.sum(D, axis=1)

        auc = roc_auc_score(labels, ano_score)
        AUC.append(auc)
        print("AUC for class {}: {}".format(normal_class, auc))

    AUC = np.array(AUC)
    print("\nAVG AUC: {}".format(AUC.mean()))


    train_features, train_labels, test_features, test_labels, \
    near_in_features, near_in_labels = read_features(aug=False, corrup=True, model=models[i], o_dim=output_dim[i])
    AUC = []
    print("Calculating AUC for Common Corruptions (C):")

    for normal_class in range(30):
        y = train_labels.copy()
        X = train_features.copy()[y == normal_class].astype("float32")

        y_test = test_labels.copy()
        X_test = test_features.copy().astype("float32")

        inl = X_test[y_test == normal_class]
        inl2 = near_in_features[near_in_labels == normal_class]
        out = X_test[y_test != normal_class]

        ############ adaptation
        X_norms = np.linalg.norm(X, axis=1)
        X = X / X_norms[:, np.newaxis]

        data = np.concatenate((inl, inl2, out)).astype("float32")
        labels = np.concatenate((np.zeros((inl.shape[0])), np.zeros((inl2.shape[0])), np.ones((out.shape[0])))).astype(int)

        data_norms = np.linalg.norm(data, axis=1)
        data = data / data_norms[:, np.newaxis]

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        D, _ = index.search(data, 1)
        ano_score = np.sum(D, axis=1)

        auc = roc_auc_score(labels, ano_score)
        AUC.append(auc)
        print("AUC for class {}: {}".format(normal_class, auc))

    AUC = np.array(AUC)
    print("\nAVG AUC: {}".format(AUC.mean()))


    AUC = []
    print("Calculating AUC for Unrealistic (U):")

    for normal_class in range(30):
        y = train_labels.copy()
        X = train_features.copy()[y == normal_class].astype("float32")

        y_test = test_labels.copy()
        X_test = test_features.copy().astype("float32")

        inl = X_test[y_test == normal_class]
        out = X_test[y_test != normal_class]

        ############ adaptation
        X_norms = np.linalg.norm(X, axis=1)
        X = X / X_norms[:, np.newaxis]

        data = np.concatenate((inl, out)).astype("float32")
        labels = np.concatenate((np.zeros((inl.shape[0])), np.ones((out.shape[0])))).astype(int)

        data_norms = np.linalg.norm(data, axis=1)
        data = data / data_norms[:, np.newaxis]

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        D, _ = index.search(data, 1)
        ano_score = np.sum(D, axis=1)

        auc = roc_auc_score(labels, ano_score)
        AUC.append(auc)
        print("AUC for class {}: {}".format(normal_class, auc))

    AUC = np.array(AUC)
    print("\nAVG AUC: {}".format(AUC.mean()))