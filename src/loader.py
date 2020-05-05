import os
import pickle

import numpy as np
import torch
import torchvision
from PIL import Image

from src.config import *

DEFAULT_TOKEN_TO_INDEX = {'<EOS>': 0, '<SOS>': 1, '<UNK>': 2}
DEFAULT_INDEX_TO_TOKEN = ['<EOS>', '<SOS>', '<UNK>']


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, formulas_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.n_samples, self.max_seq_len, self.vocab_size = 0, 0, 0
        self.index_to_token = DEFAULT_INDEX_TO_TOKEN
        self.token_to_index = DEFAULT_TOKEN_TO_INDEX
        self.vocab_size = len(self.index_to_token)

        # compute n_samples, max_seq_len, vocab_size and index of each word
        with open(formulas_path, 'r') as file:
            for line in file:
                self.n_samples += 1
                line = line.strip().split(' ')
                if len(line) > self.max_seq_len:
                    self.max_seq_len = len(line)

                for token in line:
                    if token not in self.token_to_index:
                        self.index_to_token.append(token)
                        self.token_to_index[token] = self.vocab_size
                        self.vocab_size += 1
        self.max_seq_len += 2

        # build matrix of formulas (index, for embedding)
        self.formulas = torch.zeros((self.n_samples, self.max_seq_len)).long()
        with open(formulas_path, 'r') as file:
            for sample_idx, line in enumerate(file):
                pos = 0
                self.formulas[sample_idx, 0] = self.token_to_index['<SOS>']
                pos += 1
                for token in line.strip().split(' '):
                    self.formulas[sample_idx, pos] = self.token_to_index.get(token)
                    pos += 1
                self.formulas[sample_idx, pos] = self.token_to_index['<EOS>']

    def __getitem__(self, index):
        img_filename = '%d.png' % index
        img = Image.open(os.path.join(self.images_dir, img_filename))
        img = np.array(img) - 128
        if self.transform is not None:
            img = self.transform(img)

        return img, self.formulas[index]

    def __len__(self):
        return self.n_samples

    def save(self, prefix):
        with open(prefix + 'train_dataset.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, formulas_path, max_seq_len, index_to_token, token_to_index,
                 transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.n_samples = sum(1 for _ in open(formulas_path))
        self.max_seq_len = max_seq_len
        self.index_to_token, self.token_to_index = index_to_token, token_to_index
        self.vocab_size = len(index_to_token)

        # build matrix of formulas (index, for embedding)
        self.formulas = torch.zeros((self.n_samples, self.max_seq_len)).long()
        with open(formulas_path, 'r') as file:
            for sample_idx, line in enumerate(file):
                pos = 0
                self.formulas[sample_idx, 0] = self.token_to_index['<SOS>']
                pos += 1
                for token in line.strip().split(' ')[:self.max_seq_len - 2]:
                    idx = self.token_to_index.get(token)
                    if idx:
                        self.formulas[sample_idx, pos] = idx
                    else:
                        self.formulas[sample_idx, pos] = token_to_index['<UNK>']
                    pos += 1
                self.formulas[sample_idx, pos] = self.token_to_index['<EOS>']

    def __getitem__(self, index):
        img_filename = '%d.png' % index
        img = Image.open(os.path.join(self.images_dir, img_filename))
        img = np.array(img) - 128
        if self.transform is not None:
            img = self.transform(img)

        return img, self.formulas[index]

    def __len__(self):
        return self.n_samples

    def save(self, prefix):
        with open(prefix + 'val_dataset.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.n_samples = len(
            [name for name in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, name))])

    def __getitem__(self, index):
        img_filename = '%d.png' % index
        img = Image.open(os.path.join(self.images_dir, img_filename))
        img = np.array(img) - 128
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.n_samples


print('Loading datasets...')

DUMPS_PREFIX = '../dataloader_dumps/'

if os.path.isfile(DUMPS_PREFIX + 'train_dataset.pickle'):
    with open(DUMPS_PREFIX + 'train_dataset.pickle', 'rb') as handle:
        train_dataset = pickle.load(handle)
else:
    train_dataset = TrainDataset(images_dir='../Dataset/images/images_train',
                                 formulas_path='../Dataset/formulas/train_formulas.txt',
                                 transform=torchvision.transforms.ToTensor())
    train_dataset.save(DUMPS_PREFIX)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=LOADER_NUM_WORKERS, shuffle=True)

if os.path.isfile(DUMPS_PREFIX + 'val_dataset.pickle'):
    with open(DUMPS_PREFIX + 'val_dataset.pickle', 'rb') as handle:
        val_dataset = pickle.load(handle)
else:
    val_dataset = ValDataset(images_dir='../Dataset/images/images_validation',
                             formulas_path='../Dataset/formulas/validation_formulas.txt',
                             max_seq_len=train_dataset.max_seq_len, index_to_token=train_dataset.index_to_token,
                             token_to_index=train_dataset.token_to_index, transform=torchvision.transforms.ToTensor())
    val_dataset.save(DUMPS_PREFIX)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, num_workers=LOADER_NUM_WORKERS, shuffle=True)

test_dataset = TestDataset(images_dir='../Dataset/images/images_test', transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=LOADER_NUM_WORKERS, shuffle=False)

print('All datasets are loaded')
