import gc
import io
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import logging
from PIL import Image
import glob
# Keep the import below for registering all model definitions
from models import ddpm
from models import ncsnv2
from models import  ncsnpp

import torchvision.utils as tvu
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import datasetsgu
import likelihood
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
# import matplotlib.pyplot as plt
from utils import save_checkpoint, restore_checkpoint
import pdb
import tqdm

def repre(config,
             workdir,
             task,
             eval_folder="representation",
             datasets_type = 'D'
             ):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """

  logging.info("Config =")
  logging.info(config)
  logging.info(">" * 80)
  

  # Create directory to eval_folder
  repre_dir = os.path.join(workdir, eval_folder)
  os.makedirs(repre_dir, exist_ok=True)


  # train_ds, eval_ds, _ ,_ = datasets.get_dataset(config, tasktype=task, evaluation=True)
  train_g, train_u, eval_g, eval_u, train_numg , train_numu, eval_numg, eval_nums = datasetsgu.get_dataset(config, evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size)
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  
  for ckpt in range(45678, 45679):
    # 9 -- 27
    # Wait if the target checkpoint doesn't exist yet
    logging.info("checkpoint: %d" % (ckpt))
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not os.path.exists(ckpt_filename):
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))

    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    
    ema.copy_to(score_model.parameters())

    
    datasets_type = datasets_type

    if datasets_type == 'D':
      print("no D")
      #  evalds = eval_ds
      #  sample_dir = os.path.join(repre_dir,'D')
      #os.makedirs(sample_dir, exist_ok=True)
    elif datasets_type == 'Dg':
       evalds = eval_g
       sample_dir = os.path.join(repre_dir,'Dg')
       os.makedirs(sample_dir, exist_ok=True)
    elif datasets_type == 'Du':
       evalds = eval_u
       sample_dir = os.path.join(repre_dir,'Du')
       os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(0, 1):
        for i, (test_x, test_y) in enumerate(tqdm.tqdm(evalds)):
            if i > 1:
              break
            else:
              
                score_model.eval()
                
                for m in range(test_x.shape[0]):
                  b = i * test_x.shape[0] + m
                  print(b)
                  x_uint = test_x[m].mul(255).byte()
                  x_uint = x_uint.cpu().numpy().transpose(1, 2, 0).astype('uint8')
                  save_path = os.path.join(sample_dir , "OriginalJPG")
                  if not os.path.exists(save_path):
                      os.makedirs(save_path,exist_ok=True)
                  save_name = os.path.join(save_path,'Original_images_' + str(b)+'.jpg')
                  plt.imsave(save_name, x_uint)
                
                
                if i < 2:
                #grid_image = make_grid(test_x, padding=2)
                  grid_image = make_grid(test_x, nrow=10, padding=2)
                  save_image(grid_image, os.path.join(sample_dir, 'Original_ckpt_{}_{}.png'.format(ckpt, i)))
                
                logging.info("Original image %d from ckpt %d saved" % (i, ckpt))

                test_x = test_x.to(config.device)

                eval_batch = scaler(test_x)
                _, latent_z, _ = likelihood_fn(score_model, eval_batch)
                # latent_z 
                # torch.Size([200, 3, 32, 32])

                # t (1e-3,1)
                # tensor = torch.full((200, 20), 0.03)
                
                t = torch.full((eval_batch.shape[0],), 0.02).to(eval_batch.device)
                # [0, 1) 映射到区间 (eps, sde.T)  (1e-5,1)
                z = torch.randn_like(eval_batch)
                
                mean, std = sde.marginal_prob(latent_z, t)
                perturbed_data = mean + std[:, None, None, None] * z

                

                x, nfe = sampling_fn(score_model, perturbed_data)

                # x = x.permute(0, 2, 3, 1).cpu().numpy()

                # plt.subplot(1, 2, 2)
                # plt.axis('off')
                # plt.imshow(image_grid(x))
                # plt.title('Reconstructed images')
                for m in range(test_x.shape[0]):
                  b = i * test_x.shape[0] + m
                  print(b)
                  x_uint = x[m].mul(255).byte()
                  x_uint = x_uint.cpu().numpy().transpose(1, 2, 0).astype('uint8')
                  save_path = os.path.join(sample_dir , "RepreJPG")
                  if not os.path.exists(save_path):
                      os.makedirs(save_path,exist_ok=True)
                  save_name = os.path.join(save_path,'Repre_images_' + str(b)+'.jpg')
                  plt.imsave(save_name, x_uint)

                
                if i < 3:
                #grid_image
                  grid_image_in = make_grid(x, nrow=10, padding=2)
                  save_image(grid_image_in, os.path.join(sample_dir, 'Reconstructed_ckpt_{}_{}.png'.format(ckpt, i)))
                logging.info("Reconstructed image %d from ckpt %d saved"  % (i, ckpt) )



    

    