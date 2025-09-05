"""Return training and evaluation/test datasets from config files."""
import os
from torch.utils.data import Subset
from data.celeba import CelebA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image
import numpy as np

import lmdb
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  image = np.array(image)

  crop = min(image.shape[0], image.shape[1])
  h, w = image.shape[0], image.shape[1]

  image = image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

  pil_image = Image.fromarray(image)
  pil_image = pil_image.resize((resolution, resolution), resample=Image.BICUBIC)

  return np.array(pil_image).astype(np.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = int(np.round(h * ratio))
  w = int(np.round(w * ratio))

  pil_image = Image.fromarray(image)
  pil_image = pil_image.resize((w, h), resample=Image.ANTIALIAS)

  return np.array(pil_image)

def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2

  pil_image = Image.fromarray(image)
  pil_image = pil_image.crop((left, top, left + size, top + size))

  return np.array(pil_image)

def logit_transform(image, lam=1e-6):
  image = lam + (1 - 2 * lam) * image
  return torch.log(image) - torch.log1p(-image)
    

def data_transform(config, X):
  X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
  #uniform_dequantization: If `True`, add uniform dequantization to images.
  # if config.data.uniform_dequantization:
  #     X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
      
  # if config.data.gaussian_dequantization:
  #     X = X + torch.randn_like(X) * 0.01
  # if config.data.rescaled:
  #     X = 2 * X - 1.0
  # elif config.data.logit_transform:
  #     X = logit_transform(X)

  # if hasattr(config, "image_mean"):
  #     return X - config.image_mean.to(X.device)[None, ...]
  return X


class CustomDataset(Dataset):
    def __init__(self, root, train, excluded_classes, transform=None, Dataset=None):
        if Dataset == "CIFAR10":
          self.original_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        elif Dataset == "MNIST":
          self.original_dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        else:
           print('no datasets type')
        self.excluded_classes = set(excluded_classes)
        self.indices = [i for i, (_, label) in enumerate(self.original_dataset) if label not in self.excluded_classes]

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        return len(self.indices)

# 设置要排除的类别
# 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、货车 
# excluded_classes_g = [1, 5]  # 例如，排除类别 0 和 2
# excluded_classes_g = [0]  # 例如，排除类别 0 和 2
# excluded_classes_u = [0, 2, 3, 4, 6, 7, 8, 9]  # 例如，排除类别 1 和 5

excluded_classes_g = [0,1,2,4,6,8]  # 例如，排除类别 0 和 2
excluded_classes_u = [3,5,7,9]  # 例如，排除类别 1 和 5

def get_dataset(config, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        
        evaluation: If `True`, fix number of epochs to 1.

    Returns:
        train_loader, eval_loader, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    dataset_list = ["CIFAR10", "CELEBA", "STL10"]
    if config.data.dataset not in dataset_list:
       logging.info("Non-handled dataset")
    
    def create_loader(dataset, batch_size, shuffle=False):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    if config.data.dataset == 'CIFAR10' or config.data.dataset == "MNIST":
      if config.data.random_flip and not evaluation:
        train_transform = transforms.Compose(
              [
                  transforms.Resize(config.data.image_size),
                  transforms.RandomHorizontalFlip(p=0.5),
                  transforms.ToTensor(),
              ]
          )
        test_transform = transforms.Compose(
              [transforms.Resize(config.data.image_size), 
              transforms.ToTensor()]
          )
      if config.data.random_flip is False or evaluation:
          train_transform = test_transform = transforms.Compose(
              [transforms.Resize(config.data.image_size), 
              transforms.ToTensor()]
          ) 
          
      train_g = CustomDataset(root=config.data.data_path, train=True, excluded_classes=excluded_classes_g, transform=train_transform, Dataset=config.data.dataset)
      eval_g = CustomDataset(root=config.data.data_path, train=False, excluded_classes=excluded_classes_g, transform=test_transform, Dataset=config.data.dataset)
      
      train_u = CustomDataset(root=config.data.data_path, train=True, excluded_classes=excluded_classes_u, transform=train_transform, Dataset=config.data.dataset)
      eval_u = CustomDataset(root=config.data.data_path, train=False, excluded_classes=excluded_classes_u, transform=test_transform, Dataset=config.data.dataset)
      # import pdb
      # pdb.set_trace()
      train_numg =len(train_g)
      train_numu = len(train_u)
      train_num = len(train_g) +len(train_u)  #40000  +  10000

      eval_numg = len(eval_g)    
      eval_nums = len(eval_g) +len(eval_u)   #8000   + 2000

      train_g_loader = create_loader(train_g, config.training.batch_size_gen, shuffle=True)
      train_u_loader = create_loader(train_u, config.training.batch_size_un, shuffle=True)

      eval_g_loader = create_loader(eval_g, config.eval.batch_size, shuffle=False)
      eval_u_loader = create_loader(eval_u, config.eval.batch_size, shuffle=False)

    
    elif config.data.dataset == "CELEBA":
        if config.data.random_flip:
          train_transform = transforms.Compose([transforms.CenterCrop(140),
                                      transforms.Resize(config.data.image_size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                  ])
        else:
          train_transform = transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(config.data.image_size),
                                      transforms.ToTensor(),
                                  ])

        test_transform=transforms.Compose([
                                        transforms.CenterCrop(140),
                                        transforms.Resize(config.data.image_size),
                                        transforms.ToTensor(),
                                    ])
        train_all =  CelebA(root=config.data.data_path, split='train', download=True, transform=train_transform)
        eval_all =  CelebA(root=config.data.data_path, split='test', download=True, transform=test_transform)
        valid_all =  CelebA(root=config.data.data_path, split='valid', download=True, transform=test_transform)
        
        
        un_index = 5   # 6:'Bangs' 

        indices_tu = []
        indices_tg = []
        for i, (_, target) in enumerate(train_all):
            # import pdb
            # pdb.set_trace()
            if target[un_index] == 1:
               indices_tu.append(i)
            else:
               indices_tg.append(i)

        dataset_tg = Subset(train_all, indices_tg)
        dataset_tu = Subset(train_all, indices_tu)


        indices_eg = []
        indices_eu = []
        for i, (_, target) in enumerate(eval_all):
            if target[un_index] == 1:
                indices_eu.append(i)
            else:
                indices_eg.append(i)

        dataset_eg = Subset(eval_all, indices_eg)
        dataset_eu = Subset(eval_all, indices_eu)

        indices_vg = []
        indices_vu = []
        for i, (_, target) in enumerate(valid_all):
            if target[un_index] == 1:
                indices_vu.append(i)
            else:
                indices_vg.append(i)


        dataset_vg = Subset(valid_all, indices_vg)
        dataset_vu = Subset(valid_all, indices_vu)

        valid_numg =len(dataset_vg)
        valid_numu = len(dataset_vu)

        train_numg =len(dataset_tg)
        train_numu = len(dataset_tu)
        train_num = len(train_all) #40000  +  10000

        eval_numg = len(dataset_eg)
        eval_numu = len(dataset_eu)   
        eval_nums = len(eval_all)  #8000   + 2000


        train_g_loader = create_loader(dataset_tg, config.training.batch_size_gen, shuffle=True)
        train_u_loader = create_loader(dataset_tu, config.training.batch_size_un, shuffle=True)

        eval_g_loader = create_loader(dataset_eg, config.eval.batch_size_gen, shuffle=False)
        eval_u_loader = create_loader(dataset_eu, config.eval.batch_size_un, shuffle=False)
    
    elif config.data.dataset == "STL10":
        if config.data.random_flip is False:
          tran_transform = transforms.Compose(
              [transforms.Resize(config.data.image_size), transforms.ToTensor()]
          )
        else:
            tran_transform = transforms.Compose(
               [   #transforms.CenterCrop(80),
                  transforms.Resize(config.data.image_size),
                  transforms.RandomHorizontalFlip(p=0.5),
                  transforms.ToTensor(),
                ]
            )
        # for STL10 use both train and test sets due to its small size
        train_dataset = datasets.STL10(
            config.data.data_path,
            split="train",
            download=True,
            transform=tran_transform,
        )
        # 5000
        test_dataset = datasets.STL10(
            config.data.data_path,
            split="test",
            download=True,
            transform=tran_transform,
        )
        # 8000
        dataset = ConcatDataset([train_dataset, test_dataset])
        
        
        un_index = 0   # 6:'Bangs' 

        indices_tu = []
        indices_tg = []
        for i, (_, target) in enumerate(dataset):
            if target != 0:
              indices_tg.append(i)
            else:
              indices_tu.append(i)

        dataset_tg = Subset(dataset, indices_tg)
        dataset_tu = Subset(dataset, indices_tu)
        
        indices_eg = []
        indices_eu = []
        for i, (_, target) in enumerate(train_dataset):
            if target != 0:
              indices_eg.append(i)
            else:
              indices_eu.append(i)

        dataset_eg = Subset(train_dataset, indices_eg)
        dataset_eu = Subset(train_dataset, indices_eu)
        
        
        train_g_loader = DataLoader(
            dataset_tg,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
      
        train_u_loader  = DataLoader(
            dataset_tu,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        eval_g_loader = DataLoader(
            dataset_eg,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        eval_u_loader = DataLoader(
            dataset_eu,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # valid_numg =len(dataset_vg)
        # valid_numu = len(dataset_vu)

        train_numg =len(dataset_tg) #11700
        train_numu = len(dataset_tu) # 1300
        train_num = len(dataset) # 13000

        eval_numg = len(dataset_eg)  # 4500
        eval_numu = len(dataset_eu)  # 500
        eval_nums = len(train_dataset)  # 5000
        # import pdb
        # pdb.set_trace()
      
    elif config.data.dataset == "LSUN3":
        if config.data.random_flip is False:
          tran_transform = transforms.Compose(
              [transforms.Resize(config.data.image_size), transforms.ToTensor()]
          )
        else:
            tran_transform = transforms.Compose(
               [   #transforms.CenterCrop(80),
                  transforms.Resize(config.data.image_size),
                  # transforms.RandomHorizontalFlip(p=0.5),
                  transforms.ToTensor(),
                ]
            )
        # for STL10 use both train and test sets due to its small size
        train_dataset = datasets.STL10(
            config.data.data_path,
            split="train",
            download=True,
            transform=tran_transform,
        )
        # 5000
        test_dataset = datasets.STL10(
            config.data.data_path,
            split="test",
            download=True,
            transform=tran_transform,
        )
        # 8000
        dataset = ConcatDataset([train_dataset, test_dataset])
        
        
        un_index = 0   # 6:'Bangs' 

        indices_tu = []
        indices_tg = []
        for i, (_, target) in enumerate(dataset):
            if target != 0:
              indices_tg.append(i)
            else:
              indices_tu.append(i)

        dataset_tg = Subset(dataset, indices_tg)
        dataset_tu = Subset(dataset, indices_tu)
        
        indices_eg = []
        indices_eu = []
        for i, (_, target) in enumerate(train_dataset):
            if target != 0:
              indices_eg.append(i)
            else:
              indices_eu.append(i)

        dataset_eg = Subset(train_dataset, indices_eg)
        dataset_eu = Subset(train_dataset, indices_eu)
        
        
        train_g_loader = DataLoader(
            dataset_tg,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
      
        train_u_loader  = DataLoader(
            dataset_tu,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        eval_g_loader = DataLoader(
            dataset_eg,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        eval_u_loader = DataLoader(
            dataset_eu,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # valid_numg =len(dataset_vg)
        # valid_numu = len(dataset_vu)

        train_numg =len(dataset_tg) #11700
        train_numu = len(dataset_tu) # 1300
        train_num = len(dataset) # 13000

        eval_numg = len(dataset_eg)  # 4500
        eval_numu = len(dataset_eu)  # 500
        eval_nums = len(train_dataset)  # 5000
        # import pdb
        # pdb.set_trace()
      


    return train_g_loader, train_u_loader, eval_g_loader, eval_u_loader, train_numg, train_numu, eval_numg, eval_nums
  
  

# **自定义 LSUN 数据集**
class LSUNSubsetDataset(Dataset):
    def __init__(self, lmdb_dirs, num_per_class=20000, transform=None):
        self.lmdb_dirs = lmdb_dirs
        self.transform = transform
        self.keys = []
        self.labels = []  # 记录类别信息

        # 遍历多个类别数据集（sheep, cow, bus）
        self.envs = [lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False) for lmdb_dir in lmdb_dirs]
        
        for label, lmdb_dir in enumerate(self.lmdb_dirs):
            env = self.envs[label]
            with env.begin(write=False) as txn:
                all_keys = [key for key in txn.cursor().iternext(values=False)]  # 获取所有 key
                selected_keys = all_keys[:num_per_class]  # 仅选取前 20,000 张图片
                self.keys.extend(selected_keys)
                self.labels.extend([label] * len(selected_keys))  # 记录类别（sheep=0, cow=1, bus=2）

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        label = self.labels[index]
        env = self.envs[label]  # 直接根据类别索引访问正确的 LMDB 环境
        
        with env.begin(write=False) as txn:
            value = txn.get(key)
            if value is None:
                raise KeyError(f"Key {key} not found in database")
            image = Image.open(io.BytesIO(value)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label  # 返回图片和类别标签