"""Return training and evaluation/test datasets from config files."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image
import numpy as np
from data.celeba import CelebA
import logging
from torch.utils.data import Subset
import io


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset,ConcatDataset
import lmdb

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
  if config.data.uniform_dequantization:
      X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
  # if config.data.gaussian_dequantization:
  #     X = X + torch.randn_like(X) * 0.01
  # if config.data.rescaled:
  #     X = 2 * X - 1.0
  # elif config.data.logit_transform:
  #     X = logit_transform(X)

  # if hasattr(config, "image_mean"):
  #     return X - config.image_mean.to(X.device)[None, ...]
  return X


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train, excluded_classes, transform=None):
        self.original_dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.excluded_classes = set(excluded_classes)
        self.indices = [i for i, (_, label) in enumerate(self.original_dataset) if label not in self.excluded_classes]

    def __getitem__(self, index):
        original_index = self.indices[index]
        return self.original_dataset[original_index]

    def __len__(self):
        return len(self.indices)


# 设置要排除的类别
excluded_classes = [1, 5]  # 例如，排除类别 0 和 2

def get_dataset(config, tasktype, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
        config: A ml_collection.ConfigDict parsed from config files.
        uniform_dequantization: If `True`, add uniform dequantization to images.
        evaluation: If `True`, fix number of epochs to 1.

    Returns:
        train_loader, eval_loader, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    dataset_list = ["CIFAR10", "CELEBA","STL10"]
    if config.data.dataset not in dataset_list:
       logging.info("Non-handled dataset")
                    
    if config.data.dataset == "CIFAR10":
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
    

    def create_loader(dataset, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


    if config.data.dataset == "CIFAR10":
      if tasktype == "no_u":
        #only dataset-g
        train_nou = CustomCIFAR10Dataset(root=config.data.data_path, train=True, excluded_classes=excluded_classes, transform=train_transform)
        eval_nou = CustomCIFAR10Dataset(root=config.data.data_path, train=False, excluded_classes=excluded_classes, transform=test_transform)
        train_num = len(train_nou)
        eval_num = len(eval_nou)
        train_loader = create_loader(train_nou, shuffle=False)
        eval_loader = create_loader(eval_nou, shuffle=False)

      else:  #task normal  all datasets
        train_all = datasets.CIFAR10(root=config.data.data_path, train=True, download=True, transform=train_transform)
        eval_all = datasets.CIFAR10(root=config.data.data_path, train=False, download=True, transform=test_transform)
        train_num = len(train_all)
        eval_num = len(eval_all)
        train_loader = create_loader(train_all, shuffle=False)
        eval_loader = create_loader(eval_all, shuffle=False)

    elif config.data.dataset == "CELEBA":
      if tasktype == "no_u":
        #only dataset-g
        train_all =  CelebA(root=config.data.data_path, split='train', target_type = "identity", download=True, transform=train_transform)
        eval_all =  CelebA(root=config.data.data_path, split='test', target_type = "identity", download=True, transform=test_transform)
        valid_all =  CelebA(root=config.data.data_path, split='valid', target_type = "identity", download=True, transform=test_transform)


        #----------------------------------
        # Attr set unlearnable
        # un_index = 5   # 6:'Bangs' 

        # # indices_tu = []
        # indices_tg = []
        # for i, (_, target) in enumerate(train_all):
        #     if target[un_index] != 1:
        #        indices_tg.append(i)
        #----------------------------------
        

        #----------------------------------
        # id set unlearnable
        un_index = 999  # 6:'Bangs' 

        # indices_tu = []
        indices_tg = []
        for i, (_, target) in enumerate(train_all):
            if target != un_index:
               indices_tg.append(i)

        dataset_tg = Subset(train_all, indices_tg)
        # dataset_tu = Subset(train_all, indices_tu)


        indices_eg = []
        # indices_eu = []
        for i, (_, target) in enumerate(eval_all):
            if target != un_index:
                indices_eg.append(i)

        dataset_eg = Subset(train_all, indices_eg)
        # dataset_eu = Subset(train_all, indices_eu)

        # valid

        # indices_vg = []
        # indices_vu = []
        # for i, (_, target) in enumerate(valid_all):
        #     if target[un_index] == 1:
        #         indices_vu.append(i)
        #     else:
        #         indices_vg.append(i)


        # dataset_vg = Subset(train_all, indices_vg)
        # dataset_vu = Subset(train_all, indices_vu)

        # valid_numg =len(dataset_vg)
        # valid_numu = len(dataset_vu)

        train_numg =len(dataset_tg)
        # train_numu = len(dataset_tu)
        # train_num = len(train_all) #40000  +  10000
        train_num = train_numg

        eval_numg = len(dataset_eg)
        # eval_numu = len(dataset_eu)   
        # eval_nums = len(eval_all)  #8000   + 2000
        eval_num = eval_numg
        # import pdb
        # pdb.set_trace()

        train_g_loader = create_loader(dataset_tg, shuffle=True)
        # train_g_loader = create_loader(dataset_tg, config.training.batch_size_gen, shuffle=True)
        # train_u_loader = create_loader(dataset_tu, config.training.batch_size_un, shuffle=True)
        train_loader = train_g_loader


        eval_g_loader = create_loader(dataset_eg, shuffle=True)
        # eval_g_loader = create_loader(dataset_eg, config.eval.batch_size_gen, shuffle=True)
        # eval_u_loader = create_loader(dataset_eu, config.eval.batch_size_un, shuffle=True)
        eval_loader = eval_g_loader

      else:  #task normal  all datasets
        train_all =  CelebA(root=config.data.data_path, split='train', target_type = "identity", download=True, transform=train_transform)
        eval_all =  CelebA(root=config.data.data_path, split='test', target_type = "identity", download=True, transform=test_transform)
        testeval_all =  CelebA(root=config.data.data_path, split='valid', target_type = "identity", download=True, transform=test_transform)
        train_num = len(train_all)
        eval_num = len(eval_all)
        testeval_num = len(testeval_all)
        train_loader = create_loader(train_all, shuffle=False)
        eval_loader = create_loader(eval_all, shuffle=False)
        # import pdb
        # pdb.set_trace()


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
        test_dataset = datasets.STL10(
            config.data.data_path,
            split="test",
            download=True,
            transform=tran_transform,
        )
        dataset = ConcatDataset([train_dataset, test_dataset])
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        eval_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        train_num = len(train_dataset)
        eval_num = len(test_dataset)
        # import pdb
        # pdb.set_trace()
    elif config.data.dataset == "LSUN3":
        # **数据预处理**
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # **LSUN 数据路径**
        lmdb_dirs = [
            "/home/jiang/home2/data/lsun/objects/sheep",  # LSUN Sheep 数据目录
            "/home/jiang/home2/data/lsun/objects/cow",    # LSUN Cow 数据目录
            "/home/jiang/home2/data/lsun/objects/bus",    # LSUN Bus 数据目录
        ]

        # **构建数据集**
        dataset = LSUNSubsetDataset(lmdb_dirs, num_per_class=20000, transform=transform)
      
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
        split_file = "/home/jiang/home2/data/lsun/objects/lsun_fixed_3_20000_splits.pt"
        splits = torch.load(split_file)
        train_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]
        
        if tasktype == "no_u": # cow label=1 u
          filtered_train_indices = [idx for idx in train_indices if dataset[idx][1] != 1]
          train_dataset_nou = Subset(dataset, filtered_train_indices)
          train_num = len(train_dataset_nou)
          # 打印检查
          print(f"原始 train 样本数: {len(train_indices)}")
          print(f"过滤后 train 样本数: {len(filtered_train_indices)}")
          
          filtered_eval_indices = [idx for idx in test_indices if dataset[idx][1] != 1]
          eval_dataset_nou = Subset(dataset, filtered_eval_indices)
          eval_num = len(eval_dataset_nou)
          # 打印检查
          print(f"原始 train 样本数: {len(test_indices)}")
          print(f"过滤后 train 样本数: {len(filtered_eval_indices)}")
          
          num_workers = min(os.cpu_count(), 4)  # 适应不同环境
          train_loader = DataLoader(train_dataset_nou, batch_size=batch_size, shuffle=True, num_workers=num_workers)
          eval_loader = DataLoader(eval_dataset_nou, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
        else:
          train_dataset = Subset(dataset, train_indices)
          train_num = len(train_dataset)
          val_dataset = Subset(dataset, val_indices)
          eval_dataset = Subset(dataset, test_indices)
          eval_num = len(eval_dataset)
          # print('val dataset num:', len(val_dataset))
          # print('test dataset num:', len(test_dataset))
          # print('train dataset num:', len(train_dataset))
          # **创建 DataLoader**
          num_workers = min(os.cpu_count(), 4)  # 适应不同环境
          train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
          val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
          eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
      
        # import pdb
        # pdb.set_trace()
    return train_loader, eval_loader, train_num, eval_num
  
  
  


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

