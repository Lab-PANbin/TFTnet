# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import scipy.io as sio
from sklearn import preprocessing
import scipy
import h5py
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from scipy.linalg import sqrtm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file
import matplotlib.pyplot as plt

DATASETS_CONFIG = {
    'Houston13': {
        'img': 'Houston13.mat',
        'gt': 'Houston13_7gt.mat',
    },
    'Houston18': {
        'img': 'Houston18.mat',
        'gt': 'Houston18_7gt.mat',
    },
    'paviaU': {
        'img': 'paviaU.mat',
        'gt': 'paviaU_7gt.mat',
    },
    'paviaC': {
        'img': 'paviaC.mat',
        'gt': 'paviaC_7gt.mat',
    },
    'shanghai': {},
    'hangzhou': {},
    'Dioni': {},
    'Loukia': {},
    'Salinas': {},
    'SalinasA': {}
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

# For SH、HZ Data
def cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1']
    data2 = total['DataCube2']
    gt1 = total['gt1']
    gt2 = total['gt2']

    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))
    data_scaler_s = preprocessing.scale(data_s)
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1], data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))
    data_scaler_t = preprocessing.scale(data_t)
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1], data2.shape[2])
    print(np.max(Data_Band_Scaler_s), np.min(Data_Band_Scaler_s))
    print(np.max(Data_Band_Scaler_t), np.min(Data_Band_Scaler_t))
    return Data_Band_Scaler_s, Data_Band_Scaler_t, gt1, gt2


def undersample_data(gt, desired_counts):
    """Undersample the data based on desired counts for each label."""
    mask = np.zeros_like(gt, dtype=np.bool)

    mask[gt == 4] = True

    for label, count in desired_counts.items():
        # Skip the 4th class (water) as we are taking all samples
        if label == 4:
            continue

        indices = np.argwhere(gt == label)
        selected_indices = indices[np.random.choice(indices.shape[0], count, replace=False)]
        for idx in selected_indices:
            mask[tuple(idx)] = True
    return mask


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder  # + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                              desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))


    if dataset_name == 'Houston13':
        # Load the image
        file = h5py.File(folder + 'Houston13.mat', 'r')
        img = file['ori_data'][:]

        img = img.transpose(1, 2, 0)

        rgb_bands = [13, 20, 33]

        gt_file = h5py.File(folder + 'Houston13_7gt.mat', 'r')
        gt = gt_file['map'][:]

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

        file.close()
        gt_file.close()

        # # Calculate the sample count for each class
        # gt = gt.astype(np.int64)
        # counts = np.bincount(gt.ravel())
        # total_samples = sum(counts)
        #
        # # Print the sample count for each class
        # print("Total samples:", total_samples)
        # for i, label in enumerate(label_values):
        #     print(f"{label}: {counts[i + 1]} samples")  # +1 because we are skipping the ignored_labels
        #
        # # If you want to print the count for ignored labels
        # for ignored in ignored_labels:
        #     print(f"Ignored label {ignored}: {counts[ignored]} samples")

        # Total samples: 200340
        # grass healthy: 345
        # stressed: 365
        # trees: 365
        # water: 285
        # residential buildings: 319
        # non-residential buildings: 408
        # road: 443
        # Ignored label_0: 197810

    elif dataset_name == 'Houston18':
        # Load the image
        file = h5py.File(folder + 'Houston18.mat', 'r')
        img = file['ori_data'][:]
        img = img.transpose(1, 2, 0)

        rgb_bands = [13, 20, 33]

        gt_file = h5py.File(folder + 'Houston18_7gt.mat', 'r')
        gt = gt_file['map'][:]

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]

        ignored_labels = [0]

        # Close the files
        file.close()
        gt_file.close()

        # Total samples: 200340
        # grass healthy: 1353
        # grass stressed: 4888
        # trees: 2766
        # water: 22
        # residential buildings: 5347
        # non-residential buildings: 32459
        # road: 6365
        # Ignored label0: 147140

    elif dataset_name == 'paviaU':
        # Load the image
        img = scipy.io.loadmat(os.path.join(folder, 'paviaU.mat'))['ori_data']

        rgb_bands = [20, 30, 30]

        gt = scipy.io.loadmat(os.path.join(folder, 'paviaU_7gt.mat'))['map']

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

        # Total samples: 207400
        # tree: 3064
        # asphalt: 6631
        # brick: 3682
        # bitumen: 1330
        # shadow: 947
        # meadow: 18649
        # bare soil: 5029
        # Ignored label 0: 168068

    elif dataset_name == 'paviaC':

        img = scipy.io.loadmat(os.path.join(folder, 'paviaC.mat'))['ori_data']

        rgb_bands = [20, 30, 30]

        gt = scipy.io.loadmat(os.path.join(folder, 'paviaC_7gt.mat'))['map']

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        ignored_labels = [0]

        # Total samples: 783640
        # tree: 7598
        # asphalt: 9248
        # brick: 2685
        # bitumen: 7287
        # shadow: 2863
        # meadow: 3090
        # bare soil: 6584
        # Ignored label0: 744285

    elif dataset_name == 'shanghai':
        file_path = './datasets/SH2HZ/DataCube.mat'
        img, _, gt, _ = cubeData(file_path)
        label_values = ["1", "2", "3"]
        rgb_bands = [20, 30, 30]
        ignored_labels = [0]
        # Total samples: 368000
        # 1(water): 123123
        # 2(land/building): 161689
        # 3(plant): 83188
        # Ignored label0: 0

    elif dataset_name == 'hangzhou':
        file_path = './datasets/SH2HZ/DataCube.mat'
        _, img, _, gt = cubeData(file_path)  # data_t_shape: (590, 230, 198)
        label_values = ["1", "2", "3"]
        rgb_bands = [20, 30, 30]
        ignored_labels = [0]
        # Total samples: 135700
        # 1(water): 18043
        # 2(land/building): 77450
        # 3(plant): 40207
        # Ignored label0: 0

    elif dataset_name == 'Dioni':
        path = './datasets/HyRANK/'
        img = scipy.io.loadmat(os.path.join(path, 'Dioni.mat'))['image_data']

        gt = scipy.io.loadmat(os.path.join(path, 'Dioni_gt_out68.mat'))['map']

        label_values = ["dense urban febric", "mineral extraction sites", "non irrigated arable land", "fruit trees",
                        "olive groves",
                        "coniferous forest", "dense sderophyllous vegetation", 'spare sderophyllous vegetation',
                        'sparcely vegatated areas', 'rocks and sand',
                        'water', 'coastal water']
        ignored_labels = [0]
        rgb_bands = [20, 30, 30]

        # Total samples: 344000
        # 1: 1262
        # 2: 204
        # 3: 614
        # 4: 150
        # 5: 1768
        # 6: 361
        # 7: 5035
        # 8: 6374
        # 9: 1754
        # 10: 492
        # 11: 1612
        # 12: 398
        # Ignored label 0: 323976

    elif dataset_name == 'Loukia':
        path = './datasets/HyRANK/'
        img = scipy.io.loadmat(os.path.join(path, 'Loukia.mat'))['image_data']

        gt = scipy.io.loadmat(os.path.join(path, 'Loukia_gt_out68.mat'))['map']

        label_values = ["dense urban febric", "mineral extraction sites", "non irrigated arable land", "fruit trees",
                        "olive groves",
                        "coniferous forest", "dense sderophyllous vegetation", 'spare sderophyllous vegetation',
                        'sparcely vegatated areas', 'rocks and sand',
                        'water', 'coastal water']
        ignored_labels = [0]
        rgb_bands = [20, 30, 30]

        # Total samples: 235305
        # 1: 206
        # 2: 54
        # 3: 426
        # 4: 79
        # 5: 1107
        # 6: 422
        # 7: 2996
        # 8: 2361
        # 9: 399
        # 10: 453
        # 11: 1393
        # 12: 421
        # Ignored label 0: 224988


    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](
            folder)


    ignored_labels = list(set(ignored_labels))

    # Normalization
    img = np.asarray(img, dtype='float32')

    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((m * n, -1))
    img = img / img.max()
    img_temp = np.sqrt(np.asarray((img ** 2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis=1)
    img_temp = img_temp.repeat(d, axis=1)
    img_temp[img_temp == 0] = 1
    img = img / img_temp
    img = np.reshape(img, (m, n, -1))

    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]

        state = np.random.get_state()
        np.random.shuffle(self.indices)
        np.random.set_state(state)
        np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]
        # print("lable_shape: ", label.shape)# (13,13)

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()

        # Add position information
        position = torch.tensor([x, y], dtype=torch.long)

        # return data, label
        return data, label, position


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label