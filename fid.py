import gc
import io
import os
import time
import numpy as np

import logging
from PIL import Image
import glob
# Keep the import below for registering all model definitions
from models import ddpm
from models import ncsnv2
from models import  ncsnpp

import tqdm
import torchvision.utils as tvu
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import datasetsgu
import likelihood
import evaluation
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
# import matplotlib.pyplot as plt
from utils import save_checkpoint, restore_checkpoint
from torchvision import datasets, transforms

import controllable_generation
import pdb
from torch.utils.data import DataLoader

def fid(config,
             workdir,
             task,
             eval_folder="FID",
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
  fid_dir = os.path.join(workdir, eval_folder)
  os.makedirs(fid_dir, exist_ok=True)

  
  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")


  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256  #False 
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # Wait if the target checkpoint doesn't exist yet 
    logging.info("checkpoint: %d" % (begin_ckpt,))
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    
    while not os.path.exists(ckpt_filename):
      if not waiting_message_printed:
          logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
          waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    # 加载 ckpt
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')

    state = restore_checkpoint(ckpt_path, state, device=config.device)
     
    ema.copy_to(score_model.parameters())

    

    # Directory to save samples. Different for each host to avoid writing conflicts
    this_sample_dir = os.path.join(fid_dir, f"ckpt_{ckpt}")
    os.makedirs(this_sample_dir, exist_ok=True)
    # resultsddpmc3_9901squ/eval/ckpt_500000/samples_0.png  or samples_0.npz


    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    # dtype('uint8')
    # (8000, 32, 32, 3)
    

    test_transform = transforms.Compose(
            [transforms.Resize(32), 
            transforms.ToTensor()]
        )
    
    
    test_dataset = datasets.CIFAR10(root="./data/cifar-10-python", train=False, download=False, transform=test_transform)
    # Inpaint
    # samples = np.load("./score_sde/premodel/normal/inpaint2/Dg/Impainted_Dg_45678.npy")
    # label_D = np.load("./SDE/score_sde/premodel/normal/inpaint2/Dg/Impainted_Dg_45678_y.npy")
    # test_dataset.data = np.array(samples)
    # test_dataset.targets = np.array(label_D)
    
    
    samples = np.load("./score_sde/premodel/normal/inpaint2/Dg/Impainted_Dg_45678.npy")
    label_D = np.load("./score_sde/premodel/normal/inpaint2/Dg/Impainted_Dg_45678_y.npy")
    test_dataset.data = np.array(samples)
    test_dataset.targets = np.array(label_D)
    
    

    test_data_loader = DataLoader(dataset=test_dataset,
                                batch_size=400,
                                shuffle=False, pin_memory=True,
                                drop_last=False, num_workers=4)

    
    for i, (test_x, test_y) in enumerate(tqdm.tqdm(test_data_loader)):
        # torch.Size([400, 3, 32, 32])
        # torch.float32
        # import pdb
        # pdb.set_trace()
        
        samples = np.clip(test_x.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # (400, 32, 32, 3)
        # dtype('uint8')

        # np.savez(os.path.join(this_sample_dir, "statistics_Dg.npz"), data=samples)

        gc.collect()
        latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
        # Force garbage collection again before returning to JAX code
        
        gc.collect()
        # # Save latent represents of the Inception network to disk or Google Cloud Storage
        
        # with open(file_path, "wb") as fout:
        #     io_buffer = io.BytesIO()
        #     np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
        #     fout.write(io_buffer.getvalue())
        
        # stats = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz")) #glob.glob 返回匹配的文件路径列表
        
        # for stat_file in stats:
        #     with open(stat_file, "rb") as fin:
        #         stat = np.load(fin)
        #         if not inceptionv3:
        #             all_logits.append(stat["logits"])
        #         all_pools.append(stat["pool_3"])

        logits=latents["logits"]
        pool_3=latents["pool_3"]

        all_logits.append(logits)
        all_pools.append(pool_3)

    if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config)
    data_pools = data_stats["pool_3"]


    from scipy.linalg import sqrtm

    # Compute FID
    mean_real = np.mean(data_pools, axis=0)
    mean_gen = np.mean(all_pools, axis=0)
    cov_real = np.cov(data_pools, rowvar=False)
    cov_gen = np.cov(all_pools, rowvar=False)

    # Calculate FID
    diff = mean_real - mean_gen
    fid = np.trace(cov_real + cov_gen - 2 * sqrtm(np.dot(cov_real, cov_gen)))


    logging.info("ckpt-%d --- FID: %.6e" % (ckpt, fid))
    report_file = os.path.join(fid_dir, f"report_{ckpt}.npz")
    with open(report_file, "wb") as f:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, fid=fid)
        f.write(io_buffer.getvalue())