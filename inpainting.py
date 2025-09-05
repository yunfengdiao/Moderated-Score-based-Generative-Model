import gc
import io
import os
import time
import numpy as np
import tqdm

import matplotlib.pyplot as plt
import logging
from PIL import Image
import glob
# Keep the import below for registering all model definitions
from models import ddpm
from models import ncsnv2
from models import  ncsnpp
from sampling import get_predictor, get_corrector

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
import controllable_generation
from datetime import datetime

import pdb


def inpaint(config,
             workdir,
             task,
             eval_folder="inpaint",
             datasets_type='D'):
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

  # import pdb
  # pdb.set_trace()
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

  
  predictor = get_predictor(config.sampling.conpredictor.lower())
  corrector = get_corrector(config.sampling.concorrector.lower())
  n_steps =  1
  pc_inpainter = controllable_generation.get_pc_inpainter(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=config.sampling.snr,
                                                        n_steps=n_steps,
                                                        probability_flow=config.sampling.probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=True)


  begin_ckpt = config.eval.begin_ckpt
  # 9  --  26
  
  for ckpt in range(45678,45679):
    print('ckpt:', ckpt)
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
       evalds = eval_ds
       sample_dir = os.path.join(repre_dir,'D')
       os.makedirs(sample_dir, exist_ok=True)
    elif datasets_type == 'Dg':
       evalds = eval_g
       sample_dir = os.path.join(repre_dir,'Dg')
       os.makedirs(sample_dir, exist_ok=True)
    elif datasets_type == 'Du':
       evalds = eval_u
       sample_dir = os.path.join(repre_dir,'Du')
       os.makedirs(sample_dir, exist_ok=True)

    x_inpaint_list = []
    y_label_list = []

    for i, (test_x, test_y) in enumerate(tqdm.tqdm(evalds)):
      # import pdb
      # pdb.set_trace()
      # x: torch.Size([500, 3, 32, 32])   y: torch.Size([500])
      if i < 1:  #17
        print("[{}] Epoch {}".format(str(datetime.now()), i))
        score_model.eval()


        if i < 1:
          #grid_image = make_grid(test_x, padding=2)
          for m in range(50):
              x_uint = test_x[m].mul(255).byte()
              x_uint = x_uint.cpu().numpy().transpose(1, 2, 0).astype('uint8')
              save_path = sample_dir + "Original"
              if not os.path.exists(save_path):
                  os.makedirs(save_path,exist_ok=True)
              save_name = os.path.join(save_path, 'Original_images_' + str(m)+ '.jpg')
              plt.imsave(save_name, x_uint)
          
          
          grid_image = make_grid(test_x, nrow=25, padding=2)
          save_image(grid_image, os.path.join(sample_dir, 'Original_ckpt_{}_{}.png'.format(ckpt, i)))
          logging.info("Original image %d from ckpt %d saved" % (i, ckpt))

        test_x = test_x.to(config.device)
        mask = torch.ones_like(test_x).to(config.device)
        # mask[:, :, :, 16:] = 0.  #右侧mask
        # mask[:, :, :16, :] = 0. 
        mask[:, :, 32:48, :] = 0. 
        # pdb.set_trace()

        if i < 1:
          mask_x = test_x * mask
          for m in range(50):
              x_uint = mask_x[m].mul(255).byte()
              x_uint = x_uint.cpu().numpy().transpose(1, 2, 0).astype('uint8')
              save_path = sample_dir + "Masked"
              if not os.path.exists(save_path):
                  os.makedirs(save_path,exist_ok=True)
              save_name = os.path.join(save_path, 'Masked_images_' + str(m)+ '.jpg')
              plt.imsave(save_name, x_uint)
          grid_image = make_grid(mask_x, nrow=25, padding=2)
          save_image(grid_image, os.path.join(sample_dir, 'Masked_ckpt_{}_{}.png'.format(ckpt, i)))
          logging.info("Masked image %d from ckpt %d saved"  % (i, ckpt))
        
        eval_batch = scaler(test_x)
        x_inpaint = pc_inpainter(score_model, eval_batch, mask)
        x_inpaint_list.append(x_inpaint)

        y_label_list.append(test_y)

        if i < 1:

          for m in range(50):
            
              x_uint = x_inpaint[m].mul(255).byte()
              x_uint = x_uint.cpu().numpy().transpose(1, 2, 0).astype('uint8')
              save_path = sample_dir + "Inpainted"
              if not os.path.exists(save_path):
                  os.makedirs(save_path,exist_ok=True)
              save_name = os.path.join(save_path, 'Inpainted_images_' + str(m)+ '.jpg')
              plt.imsave(save_name, x_uint)
              
          grid_image = make_grid(x_inpaint, nrow=25, padding=2)
          save_image(grid_image, os.path.join(sample_dir, 'Impainted_{}.png'.format(i)))
          logging.info("Impainted image saved")
      
      
      
      else:
        break
      logging.info("Inpainting image %d from ckpt %d"  % (i, ckpt))
          
    
    x_inpainted_list = torch.cat(x_inpaint_list, 0)
    x_inpainted = x_inpainted_list.mul(255).byte()
    x_inpainted = x_inpainted.cpu().numpy().transpose(0, 2, 3, 1)

    y_label_list = torch.cat(y_label_list, 0)
    y_label = y_label_list.cpu().numpy()

    save_name = os.path.join(sample_dir, 'Inpainted' + "_" + str(datasets_type) + "_" + str(ckpt) + '.npy')
    np.save(save_name, x_inpainted)


    save_name_y = os.path.join(sample_dir, 'Inpainted' + "_" + str(datasets_type) + "_" + str(ckpt) +   "_" + 'y' + '.npy')
    np.save(save_name_y, y_label)


          

            


                



    

    