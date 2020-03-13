from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import pandas as pd
from skimage.io import imread
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from .utils import download_url, check_integrity, noisify

class ImageWoof(data.Dataset):
    """`ImageWoof-320 <https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz> _Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'imagewoof2-320'
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz" 
    filename = "imagewoof2-320.tgz"
    tgz_md5 = 'af65be7963816efa949fa3c3b4947740'

    def __init__(self, root, train=True, csv_file=None,
                 transform=None, target_transform=None,
                 download=False, 
                 noise_type='clean', noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='imagewoof2-320'
        self.noise_type=noise_type
        self.nb_classes=10
    self.class_map = {
                         'n02086240':0,
                         'n02096294':1,
                         'n02089973':2,
                         'n02111889':3,
                         'n02115641':4,
                         'n02105641':5,
                         'n02087394':6,
                         'n02088364':7,
                         'n02099601':8,
                         'n02093754':9
    }

        if download:
            self.download()

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            train_labels = []
            if csv_file is None:
                path = os.path.join(self.root, self.dataset, 'train')
                classes = os.listdir(path) 
                for cls in classes:
                    pth = os.path.join(path, cls)
                    files = os.listdir(pth)
                    file_paths = [os.path.join(pth, f) for f in files]
                    self.train_data += files
                    train_labels += [str(pth).split('/')[-1]]*len(files)
                    self.train_labels = [self.class_map[i] for i in train_labels]

            else:
                csv_file_path = os.path.join(self.root, csv_file)
                train_df = pd.read_csv(csv_file_path)
                self.train_data = train_df.files
                self.train_labels = train_df.label

            #if noise_type is not None:
            if noise_type !='clean':
                # noisify train data
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
        else:
            if csv_file in None:
                self.test_data = []
                test_labels = []
                path = Path(os.path.join(self.root, self.dataset, 'val'))
                classes = os.listdir(path) 
                for cls in classes:
                    pth = os.path.join(path, cls)
                    files = os.listdir(pth)
                    file_paths = [os.path.join(pth, f) for f in files]
                    self.test_data += files
                    test_labels += [str(pth).split('/')[-1]]*len(files)
                    self.test_labels = [self.class_map[i] for i in test_labels]

            else:
                csv_file_path = os.path.join(self.root, csv_file)
                test_df = pd.read_csv(csv_file_path)
                self.test_data = test_df.files
                self.test_labels = test_df.label



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = imread(self.train_data[index]), self.train_noisy_labels[index]
            else:
                img, target = imread(self.train_data[index]), self.train_labels[index]
        else:
            img, target = imread(self.test_data[index]), self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        filename = self.filename
        fpath = os.path.join(root, self.base_folder, filename)
        if not check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

