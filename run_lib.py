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
from models.emaddpm import EMAHelper
import matplotlib.pyplot as plt

import torchvision.utils as tvu
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import datasetsgu
import evaluation
import likelihood
import sde_lib
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from utils import save_checkpoint, restore_checkpoint
import pdb


def train(config, workdir, task, finetune = False):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  logging.info("Config =")
  logging.info(">" * 80)
  logging.info(config)
  logging.info("<" * 80)

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)
  
  sample0_dir = os.path.join(workdir, "samples0")
  os.makedirs(sample0_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  os.makedirs(tb_dir, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, epoch=0, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
  # Resume training when intermediate checkpoints are detected
  # import pdb
  # pdb.set_trace()
  
  
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  
  initial_epoch = int(state['epoch'])
  initial_step = int(state['step'])

  

  # Build data iterators
  train_loader, eval_loader, train_nums, eval_nums = datasets.get_dataset(config, tasktype=task)
  
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

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

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  losstype = task
  # import pdb
  # pdb.set_trace()
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, loss_type=losstype)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, loss_type=losstype)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters
  training_epochs = int(num_train_steps * config.training.batch_size / train_nums)
  logging.info("train_nums: %d, batch size: %d" % (train_nums, config.training.batch_size ))
  logging.info("eval_nums: %d, batch size: %d" % (eval_nums, config.training.batch_size ))
  logging.info("an epo contains %d steps.training n_iters %d. training_epochs: %d. " % ((train_nums/config.training.batch_size), num_train_steps, training_epochs))
  #an epo contains 390 steps.training n_iters 1300001. training_epochs: 3328.

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d of epoch %d." % (initial_step, initial_epoch))

  #for step in range(initial_step, num_train_steps + 1):
  start_epoch = initial_epoch
  step = initial_step
  data_time = 0
  data_start = time.time()

  for epoch in range(start_epoch, training_epochs + 1):
    # if epoch > 0 :
    #   import pdb
    #   pdb.set_trace()
    #   break
    
    # y_all = []
    for i, (x, y) in enumerate(train_loader):
      # if i == 0:
      #   for m in range(10):
      #     x_uint = x[m].mul(255).byte()
      #     x_uint = x_uint.cpu().numpy().transpose(1, 2, 0)#.astype('uint8')
      #     save_name = os.path.join(sample0_dir, str(i)+'_'+str(m)+'.jpg')
      #     plt.imsave(save_name, x_uint)
      # import pdb
      # pdb.set_trace()
      #   print(i)
      #   print(x.shape)
      #   y = y
      #   y_all.append(y)
      # y_all = torch.cat(y_all, 0)
      # # 指定要查找的数
      # for i in range(1, 10178):

      #   # 使用torch.eq()函数来创建一个布尔张量，表示张量中与指定数相等的位置
      #   equal_mask = torch.eq(y_all, i)

      #   # 使用torch.sum()函数来计算布尔张量中True值的数量，即指定数出现的次数
      #   count = torch.sum(equal_mask).item()
      #   logging.info(f"The number {i} appears {count} times in the tensor.")  
    
      
      if epoch == start_epoch and i < (step % int(train_nums/config.training.batch_size+1)):
          pass
      else:
        # n = x.size(0)
        score_model.train()
        step += 1

        x = x.to(config.device)
        if config.data.uniform_dequantization:
          x = datasets.data_transform(config, x)

        # Execute one training step
        train_batch = scaler(x)
        loss = train_step_fn(state, train_batch)
        
        logging.info("step: %s, iter: %s, epoch: %s" % (step, i, epoch)) 


        #50  every 50  debug 1
        if step % config.training.log_freq == 0:  
          logging.info("step: %d, epoch: %d, training_loss: %.5e." % (step, epoch, loss.item()))
          writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        #save evety 10000 to resume training
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
          state['epoch'] = epoch
          save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        #100   eval
        if step % config.training.eval_freq == 0:
          for j, (xx, yy) in enumerate(eval_loader):
            if j == 0 :
              score_model.eval()
              xx = xx.to(config.device)
              eval_batch = scaler(xx)
              eval_loss = eval_step_fn(state, eval_batch)

              logging.info("step: %d, epoch: %d, eval_loss: %.5e." % (step, epoch, eval_loss.item() ))
              writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        # Save the checkpoint 50000 per steps 
        # debug 5
        if step != 0 and step % config.training.snapshot_freq == 0 or \
           step < config.training.snapshot_freq and step % 200 == 0 or \
           step == num_train_steps:
          
          save_step = step // config.training.snapshot_freq
          save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

          # Generate and save samples
          if config.training.snapshot_sampling:
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            sample, n = sampling_fn(score_model)
            #sample dtype  torch.float32
            #shape         torch.Size([128, 3, 32, 32])
            #n   2000
            ema.restore(score_model.parameters())

            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            os.makedirs(this_sample_dir, exist_ok=True)

            # nrow = int(np.sqrt(sample.shape[0])) # 128   11
            grid_image = make_grid(sample, nrow=8, padding=2)

            # plt.imshow(grid_image.permute(1, 2, 0).cpu().numpy())
            # plt.axis('off')
            # plt.show()
            save_image(grid_image, os.path.join(this_sample_dir, 'samples_{}.png'.format(step)))

            # 保存 NumPy 数组
            sample = np.clip(sample.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            #dtype('uint8')   (128, 32, 32, 3)
            sample_path = os.path.join(this_sample_dir, "sample.np")
            with open(sample_path, "wb") as fout:
              np.save(fout, sample)
            
    data_time += time.time() - data_start
    data_start = time.time()

    logging.info("Average time spent on an epo: %5e." % (data_time / (epoch + 1)))

def evaluate(config,
             workdir,
             task,
             eval_folder="eval"):
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
  eval_dir = os.path.join(workdir, eval_folder)
  os.makedirs(eval_dir, exist_ok=True)

  # Build data pipeline
  # uniform_de  False
  train_ds, eval_ds, _ ,_ = datasets.get_dataset(config, tasktype=task, evaluation=True)

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

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting,
                                   loss_type=None)

  # Create data loaders for likelihood evaluation. 
  # Only evaluate on uniformly dequantized data. 
  # uniform_dequantization=True
  # <<<<<<<<<<|evaluate|>>>>>>>>>>>
  train_ds_bpd, eval_ds_bpd, _ , _ = datasets.get_dataset(config, tasktype='normal', evaluation=True)
  # ALL train dataset
  # ALL test dataset
  # uniform_dequantization = True
  if config.eval.bpd_dataset.lower() == 'train':
    ds_bpd = train_ds_bpd 
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_bpd = eval_ds_bpd
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256 #inceptionv3 = False 32
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
  #get inception model according image size

  begin_ckpt = config.eval.begin_ckpt
  # 9  --  26
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    # 9 -- 27
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    while not os.path.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)

    # try:
    #   state = restore_checkpoint(ckpt_path, state, device=config.device)
    # except:
    #   time.sleep(60)
    #   try:
    #     state = restore_checkpoint(ckpt_path, state, device=config.device)
    #   except:
    #     time.sleep(120)
    #     state = restore_checkpoint(ckpt_path, state, device=config.device)
        
    # state = restore_checkpoint(ckpt_path, state, device=config.device)
  
    # ema.copy_to(score_model.parameters())

    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      for epoch in range(0, 1):
        for i, (test_x, test_y) in enumerate(eval_ds):
          #len(eval_ds)  9.7656 =  10000 / batch_size=1024
          #1024, 3, 32, 32

          score_model.eval()
          test_x = test_x.to(config.device)

          eval_batch = scaler(test_x)
          eval_loss = eval_step(state, eval_batch)

          logging.info("ckpt: %d, iter: %d, eval_loss: %.5e." % (ckpt, i, eval_loss.item() ))
          # writer.add_scalar("eval_loss", eval_loss.item(), step)
          
          all_losses.append(eval_loss.item())
          if (i + 1) % 5 == 0:
            logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)

      # 保存 NumPy 数组为压缩文件
      file_path = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
      with open(file_path, "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      train_g_bpd, train_u_bpd, eval_g_bpd, eval_u_bpd, train_numg , train_numu, eval_numg, eval_nums = datasetsgu.get_dataset(config, evaluation=True)
      
      
      # Test --> M bpd
      bpds = []

      ############################################
      #  test datasets bpd on model_all/model_gen

      test_all = True

      if test_all:
        if task == 'normal':
          logging.info(">" * 80)
          logging.info("test datasets bpd on model_all")
        else:
          logging.info(">" * 80)
          logging.info("test datasets bpd on model_gen")
        # for repeat in range(bpd_num_repeats):
        for repeat in range(1):
          for batch_id, (batch_x, batch_y) in enumerate(ds_bpd):
            score_model.eval()
            batch_x  = batch_x.to(config.device)
            #uniform_dequantization = True
            #logging.info("uniform_dequantization: %s" % uniform_dequantization)
            logging.info("uniform_dequantization")
            eval_batch = datasets.data_transform(config, batch_x)

            eval_batch = scaler(eval_batch)
            bpd = likelihood_fn(score_model, eval_batch)[0]
            bpd = bpd.detach().cpu().numpy().reshape(-1)
            bpds.extend(bpd)
            logging.info("ckpt: %d, repeat: %d, batch: %d, all test mean bpd : %6f" \
                        % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
            bpd_round_id = batch_id + len(ds_bpd) * repeat
            # i + 33 * repeat
            # Save bits/dim to disk or Google Cloud Storage
            file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz")
            with open(file_path, "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, bpd)
              fout.write(io_buffer.getvalue())


      ############################################
      #  test u bpd on model_all/model_gen

      if task == 'normal':
        logging.info(">" * 80)
        logging.info("test u bpd on model_all")
      else:
        logging.info(">" * 80)
        logging.info("test u bpd on model_gen")
      bpdsum = []
      
      #for repeat in range(bpd_num_repeats):
      for repeat in range(1):
        for batch_id, (batch_xu, batch_yu) in enumerate(eval_u_bpd):
          score_model.eval()
          batch_xu  = batch_xu.to(config.device)
          #uniform_dequantization = True
          #logging.info("uniform_dequantization: %s" % uniform_dequantization)
          logging.info("uniform_dequantization")
          eval_batchu = datasets.data_transform(config, batch_xu)

          eval_batchu = scaler(eval_batchu)
          bpdum = likelihood_fn(score_model, eval_batchu)[0]
          bpdum = bpdum.detach().cpu().numpy().reshape(-1)
          bpdsum.extend(bpdum)
          logging.info("ckpt: %d, repeat: %d, batch: %d, test u mean bpd: %6f" \
                      % (ckpt, repeat, batch_id, np.mean(np.asarray(bpdsum))))
          bpd_round_id = batch_id + len(eval_u_bpd) * repeat
          # i + 33 * repeat
          # Save bits/dim to disk or Google Cloud Storage
          file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_um.npz")
          with open(file_path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpdsum)
            fout.write(io_buffer.getvalue())

      ############################################
      #  test u bpd on model_all/model_gen

      if task == 'normal':
        logging.info(">" * 80)
        logging.info("test g bpd on model_all")
      else:
        logging.info(">" * 80)
        logging.info("test g bpd on model_gen")
      bpdsgm = []
      #for repeat in range(bpd_num_repeats):
      for repeat in range(1):
        for batch_id, (batch_xg, batch_yg) in enumerate(eval_g_bpd):
          score_model.eval()
          batch_xg  = batch_xg.to(config.device)
          #uniform_dequantization = True
          #logging.info("uniform_dequantization: %s" % uniform_dequantization)
          logging.info("uniform_dequantization")
          eval_batchg = datasets.data_transform(config, batch_xg)

          eval_batchg = scaler(eval_batchg)
          bpdgm = likelihood_fn(score_model, eval_batchg)[0]
          bpdgm = bpdgm.detach().cpu().numpy().reshape(-1)
          bpdsgm.extend(bpdgm)
          logging.info("ckpt: %d, repeat: %d, batch: %d, test g mean bpd: %6f" \
                      % (ckpt, repeat, batch_id, np.mean(np.asarray(bpdsgm))))
          bpd_round_id = batch_id + len(eval_g_bpd) * repeat
          # i + 33 * repeat
          # Save bits/dim to disk or Google Cloud Storage
          file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_gm.npz")
          with open(file_path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpdsgm)
            fout.write(io_buffer.getvalue())
        

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      #===================================================#
      ######### generate samples and save samples #########
      #===================================================#
      need_samples = True
      if need_samples:
        print("sampling")
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
        #50000 //  1024   +1
        for r in range(num_sampling_rounds):

          # Directory to save samples. Different for each host to avoid writing conflicts
          this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
          os.makedirs(this_sample_dir, exist_ok=True)

          samples, n = sampling_fn(score_model)
          
          # sample_dir = os.path.join(this_sample_dir, "iter_{}".format(r))
          # os.makedirs(sample_dir, exist_ok=True)

          # nrow = int(np.sqrt(sample.shape[0])) # 128   11
          grid_image = make_grid(samples, nrow=32, padding=2)
          # plt.imshow(grid_image.permute(1, 2, 0).cpu().numpy())
          # plt.axis('off')
          # plt.show()
          save_image(grid_image, os.path.join(this_sample_dir, 'samples_{}.png'.format(r)))
          logging.info("sampling -- ckpt: %d, round: %d / %d" % (ckpt, r, num_sampling_rounds-1))

          samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
          samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
          # Write samples to disk or Google Cloud Storage
          file_path = os.path.join(this_sample_dir, f"samples_{r}.npz")
          with open(file_path, "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, samples=samples)
              fout.write(io_buffer.getvalue())


          # Force garbage collection before calling TensorFlow code for Inception network
          gc.collect()
          #<<<<<<<<<<<<||>>>>>>>>>>>>>>>>>
          latents = evaluation.run_inception_distributed(samples, inception_model, inceptionv3=inceptionv3)
          # Force garbage collection again before returning to JAX code
          gc.collect()
          
          # Save latent represents of the Inception network to disk or Google Cloud Storage
          file_path = os.path.join(this_sample_dir, f"statistics_{r}.npz")
          with open(file_path, "wb") as fout:
              io_buffer = io.BytesIO()
              np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
              fout.write(io_buffer.getvalue())
        
        #===================================================#
        ######### generate samples and save samples #########
        #===================================================#

      # Compute inception scores, FIDs and KIDs.
      # Load all statistics that have been previously computed and saved for each host
      all_logits = []
      all_pools = []
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")  #eval/ckpt_*/statistics_* 
      stats = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz")) #获取目录下所有 statistics_* 文件
      for stat_file in stats:
        with open(stat_file, "rb") as fin:
          stat = np.load(fin)
          if not inceptionv3:
              all_logits.append(stat["logits"])
          all_pools.append(stat["pool_3"])

      if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
      all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]


      # Load pre-computed dataset statistics.
      data_stats = evaluation.load_dataset_stats(config)
      import pdb
      pdb.set_trace()
      data_pools = data_stats["pool_3"]

      # # Compute FID/KID/IS on all samples together.
      # Compute IS
      # if not inceptionv3:
      #     # Compute Inception Score

      #     # Calculate probabilities
      #     preds = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)

      #     # Calculate Inception Score
      #     kl_divs = preds * (np.log(preds) - np.log(np.mean(preds, axis=0, keepdims=True)))
      #     kl_divergence = np.mean(np.sum(kl_divs, axis=1))
      #     inception_score = np.exp(kl_divergence)
      # else:
      #     inception_score = -1

      from scipy.linalg import sqrtm

      # Compute FID
      mean_real = np.mean(data_pools, axis=0)
      mean_gen = np.mean(all_pools, axis=0)
      cov_real = np.cov(data_pools, rowvar=False)
      cov_gen = np.cov(all_pools, rowvar=False)

      # Calculate FID
      diff = mean_real - mean_gen
      fid = np.trace(cov_real + cov_gen - 2 * sqrtm(np.dot(cov_real, cov_gen)))

      # Compute KID
      # dot_product = np.dot(data_pools, np.transpose(all_pools))
      # dot_product_exp = np.exp(np.sum(dot_product, axis=1, keepdims=True) / len(all_pools))
      # kid = np.mean(dot_product_exp)

      # # Cleaning up
      # del dot_product, dot_product_exp

      # logging.info("ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" \
      #              % (ckpt, inception_score, fid, kid))

      # # Saving the report
      # report_file = os.path.join(eval_dir, f"report_{ckpt}.npz")
      # with open(report_file, "wb") as f:
      #     io_buffer = io.BytesIO()
      #     np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      #     f.write(io_buffer.getvalue())

      logging.info("ckpt-%d --- FID: %.6e"  % (ckpt, fid))

      # Saving the report
      report_file = os.path.join(eval_dir, f"report_{ckpt}.npz")
      with open(report_file, "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, fid=fid)
          f.write(io_buffer.getvalue())
