import gc
import io
import os
import time
import numpy as np
from itertools import cycle

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
import datasetsgu
import datasets
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
  if finetune:
    checkpoint_pretrain_dir = './score_sde/premodel/normal/checkpoints/checkpoint_45678.pth'
    logging.info("checkpoint_pretrain_dir: %s" % (checkpoint_pretrain_dir))
    state = restore_checkpoint(checkpoint_pretrain_dir, state, config.device)
    initial_epoch = int(0)
    initial_step = int(0)
    
  else:
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_epoch = int(state['epoch'])
    initial_step = int(state['step'])


  # Build data iterators
  train_g_loader, train_u_loader, eval_g_loader, eval_u_loader, train_numg , train_numu, eval_numg, eval_nums = datasetsgu.get_dataset(config)
  
  # Create data normalizer and its inverse
  scaler = datasetsgu.get_data_scaler(config)
  inverse_scaler = datasetsgu.get_data_inverse_scaler(config)

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
  
  train_step_fn = losses.get_step_fngu(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting, loss_type=losstype)
  eval_step_fn = losses.get_step_fngu(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, loss_type=losstype)
  

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (32, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters
  training_epochs = int(num_train_steps * config.training.batch_size / train_numg)
  logging.info("train g nums: %d, train u nums: %d, batch size: %d" % (train_numg, train_numu, config.training.batch_size ))
  logging.info("eval_nums: %d, batch size: %d" % (eval_nums, config.training.batch_size ))
  logging.info("an epo contains %d steps. training n_iters %d. training_epochs: %d. " 
               % ((train_numg/config.training.batch_size), num_train_steps, training_epochs))
  #an epo contains 390 steps.training n_iters 1300001. training_epochs: 3328.

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d of epoch %d." % (initial_step, initial_epoch))
  
  #for step in range(initial_step, num_train_steps + 1):
  start_epoch = initial_epoch
  step = initial_step
  data_time = 0
  data_start = time.time()

  
  for epoch in range(start_epoch, training_epochs + 1):
    for i, (xg, yg), (xu, yu) in zip(range(int(train_numg/config.training.batch_size)), cycle(train_g_loader), cycle(train_u_loader)):
      pdb.set_trace()
      if epoch == start_epoch and i < (step % int(train_numg/config.training.batch_size+1)):
          pass
      else:
        # n = x.size(0)
        score_model.train()
        step += 1
        
        xg = xg.to(config.device) # torch.Size([128, 3, 32, 32]) yg 
        xu = xu.to(config.device) # yu:1, 5

        if config.data.uniform_dequantization:
          xg = datasetsgu.data_transform(config, xg)
          xu = datasetsgu.data_transform(config, xu)

        # Execute one training step
        train_g_batch = scaler(xg)
        train_u_batch = scaler(xu)

        loss, lossg, lossu = train_step_fn(state, train_g_batch, train_u_batch)
        
        print("step:", step, "iter:", i, "epoch:", epoch, 
              "loss:", loss.item(), "lossg:", lossg.item(), "lossu:", lossu.item() )

        #50  every 50  debug 1
        if step % config.training.log_freq == 0:  
          logging.info("step: %d, epoch: %d, training_loss: %.5e, training_lossg: %.5e., training_lossu: %.5e." 
                       % (step, epoch, loss.item(), lossg.item(), lossu.item()))
          writer.add_scalar("training_loss", loss, step)
          writer.add_scalar("training_lossg", lossg, step)
          writer.add_scalar("training_lossu", lossu, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        #save evety 10000 to resume training
        #debug 2
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
          state['epoch'] = epoch
          save_checkpoint(checkpoint_meta_dir, state)
    
        # Report the loss on an evaluation dataset periodically
        #100   eval
        if step % config.training.eval_freq == 0:
          for j, (xxg, yyg), (xxu, yyu) in zip(range(int(eval_numg/config.eval.batch_size_gen)), cycle(eval_g_loader), cycle(eval_u_loader)):
            if j == 0 :

              score_model.eval()
              xxg = xxg.to(config.device)
              xxu = xxu.to(config.device)

              eval_g_batch = scaler(xxg)
              eval_u_batch = scaler(xxu)

              eval_loss, eval_lossg, eval_lossu = eval_step_fn(state, eval_g_batch, eval_u_batch)
              
              logging.info("step: %d, epoch: %d, eval_loss: %.5e, eval_lossg:%.5e, eval_lossu:%.5e" 
                           % (step, epoch, eval_loss.item(), eval_lossg.item(), eval_lossu.item() ))
              
              
              writer.add_scalar("eval_loss", eval_loss.item(), step)
              writer.add_scalar("eval_lossg", eval_lossg.item(), step)
              writer.add_scalar("eval_lossu", eval_lossu.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        # Save the checkpoint 50000 per steps 
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


  # Create data normalizer and its inverse
  scaler = datasetsgu.get_data_scaler(config)
  inverse_scaler = datasetsgu.get_data_inverse_scaler(config)

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
  
   # Build data pipeline  # uniform_de False
  train_g_ds, train_u_ds, eval_g_ds, eval_u_ds, train_numg , train_numu, eval_numg, eval_nums = datasetsgu.get_dataset(config, evaluation=True)
  # train_ds, eval_ds, _, _ = datasetsgu.get_dataset(config, evaluation=True)

  #————————————————————————————————————————————————————————————————————————————————————————————————
  # Create the one-step evaluation function when loss computation is enabled
  ###<<<<<<<<<<<<<<<<|eval loss|>>>>>>>>>>>>>>>>>
 
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step_fn = losses.get_step_fngu(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting, loss_type=task)

  #————————————————————————————————————————————————————————————————————————————————————————————————
  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  # <<<<<<|evaluate|>>>>>>>>
  ####<<<<<<<<<<<<<<<<|NLL bpd|>>>>>>>>>>>>>>>>>
  # bpd datasets   # uniform_dequantization = True  计算loss 不用true
  if config.eval.bpd_dataset.lower() == 'train':
    ds_g_bpd = train_g_ds
    ds_u_bpd = train_u_ds
    bpd_num_repeats = 1
  elif config.eval.bpd_dataset.lower() == 'test':
    # Go over the dataset 5 times when computing likelihood on the test dataset
    ds_g_bpd = eval_g_ds
    ds_u_bpd = eval_u_ds
    bpd_num_repeats = 5
  else:
    raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
  
  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)
  #————————————————————————————————————————————————————————————————————————————————————————————————

  # Sampling  caculate FID
  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size_sampling,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Use inceptionV3 for images with resolution higher than 256.
  inceptionv3 = config.data.image_size >= 256  #False 
  inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    logging.info("checkpoint: %d" % (ckpt,))
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
    # while not os.path.exists(ckpt_filename):
    #   if not waiting_message_printed:
    #       logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
    #       waiting_message_printed = True
    #   time.sleep(60)

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    # 加载 ckpt
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

    ema.copy_to(score_model.parameters())
    logging.info("loaded checkpoint: %d" % (ckpt,))
    

    #————————————————————————————————————————————————————————————————————————————————————————————————
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      all_losses = []
      all_lossesg = []
      all_lossesu = []
      #for epoch in range(0, 1):
      for i, (xg, yg), (xu, yu) in zip(range(int(eval_numg/config.eval.batch_size_gen)), cycle(eval_g_ds), cycle(eval_u_ds)):
      # for i, (xg, yg), (xu, yu) in zip(eval_g_ds, eval_u_ds):
        score_model.eval()
        xg = xg.to(config.device)
        xu = xu.to(config.device)

        eval_g_batch = scaler(xg)
        eval_u_batch = scaler(xu)

        eval_loss, eval_lossg, eval_lossu = eval_step_fn(state, eval_g_batch, eval_u_batch)

        logging.info("ckpt: %d, iter: %d, eval_loss: %.5e., eval_loss_gen: %.5e., eval_loss_un: %.5e." \
                     % (ckpt, i, eval_loss.item(), eval_lossg.item(), eval_lossu.item()  ))
          # writer.add_scalar("eval_loss", eval_loss.item(), step)
        all_losses.append(eval_loss.item())
        all_lossesg.append(eval_lossg.item())
        all_lossesg.append(eval_lossu.item())
        if (i + 1) % 5 == 0:
            logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      all_lossesg = np.asarray(all_lossesg)
      all_lossesu = np.asarray(all_lossesu)

      # 保存 NumPy 数组为压缩文件
      file_path = os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz")
      with open(file_path, "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        np.savez_compressed(io_buffer, all_lossesg=all_lossesg, mean_lossg=all_lossesg.mean())
        np.savez_compressed(io_buffer, all_lossesu=all_lossesu, mean_lossu=all_lossesu.mean())
        fout.write(io_buffer.getvalue())
    
    #————————————————————————————————————————————————————————————————————————————————————————————————
    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      train_ds_bpd, eval_ds_bpd, _ , _ = datasets.get_dataset(config, tasktype='normal', evaluation=True)
      bpdggs = []
      bpduus = []
      bpds = []

      logging.info(">" * 80)
      logging.info("test bpd on model_3square") 
      for repeat in range(1):
        for batch_idt, (batch_x, batch_y) in enumerate(eval_ds_bpd):
          score_model.eval()
          batch_x = batch_x.to(config.device)
          #uniform_dequantization = True
          #logging.info("uniform_dequantization: %s" % uniform_dequantization)
          logging.info("uniform_dequantization")
          eval_batch = datasets.data_transform(config, batch_x)

          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info("ckpt: %d, repeat: %d, batch: %d, all test mean bpd : %6f" \
                      % (ckpt, repeat, batch_idt, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_idt + len(eval_ds_bpd) * repeat
          # i + 33 * repeat
          # Save bits/dim to disk or Google Cloud Storage
          file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_test.npz")
          with open(file_path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

      # logging.info(">" * 80)
      # logging.info("test g bpd on model_3square") 
      # for repeat in range(1):
      #   for batch_idg, (batch_xg, batch_y) in enumerate(ds_g_bpd):
      #     score_model.eval()
      #     batch_xg = batch_xg.to(config.device)
      #     #uniform_dequantization = True
      #     #logging.info("uniform_dequantization: %s" % uniform_dequantization)
      #     logging.info("uniform_dequantization")
      #     eval_batchg = datasets.data_transform(config, batch_xg)

      #     eval_batchg = scaler(eval_batchg)
      #     bpdgg = likelihood_fn(score_model, eval_batchg)[0]
      #     bpdgg = bpdgg.detach().cpu().numpy().reshape(-1)
      #     bpdggs.extend(bpdgg)
      #     logging.info("ckpt: %d, repeat: %d, batch: %d, mean bpd_gen : %6f" \
      #                 % (ckpt, repeat, batch_idg, np.mean(np.asarray(bpdggs))))
      #     bpd_round_id = batch_idg + len(ds_g_bpd) * repeat
      #     # i + 33 * repeat
      #     # Save bits/dim to disk or Google Cloud Storage
      #     file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_gen.npz")
      #     with open(file_path, "wb") as fout:
      #       io_buffer = io.BytesIO()
      #       np.savez_compressed(io_buffer, bpdggs)
      #       fout.write(io_buffer.getvalue())


      # logging.info(">" * 80)
      # logging.info("test u bpd on model_3square") 
      # for repeat in range(1):
      #   for batch_idu, (batch_xu, batch_y) in enumerate(ds_u_bpd):
      #     score_model.eval()
      #     batch_xu = batch_xu.to(config.device)
      #     #uniform_dequantization = True
      #     #logging.info("uniform_dequantization: %s" % uniform_dequantization)
      #     logging.info("uniform_dequantization")
      #     eval_batchu = datasets.data_transform(config, batch_xu)

      #     eval_batchu = scaler(eval_batchu)
      #     bpduu = likelihood_fn(score_model, eval_batchu)[0]
      #     bpduu = bpduu.detach().cpu().numpy().reshape(-1)
      #     bpduus.extend(bpduu)
      #     logging.info("ckpt: %d, repeat: %d, batch: %d, mean bpd_un : %6f" % (ckpt, repeat, batch_idu, np.mean(np.asarray(bpduus))))
      #     bpd_round_id = batch_idu + len(ds_u_bpd) * repeat
      #     # i + 33 * repeat
      #     # Save bits/dim to disk or Google Cloud Storage
      #     file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}_un.npz")
      #     with open(file_path, "wb") as fout:
      #       io_buffer = io.BytesIO()
      #       np.savez_compressed(io_buffer, bpduus)
      #       fout.write(io_buffer.getvalue())

      ###### for repeat in range(bpd_num_repeats):
      for repeat in range(1):
        for batch_id, (batch_xg, batch_yg), (batch_xu, batch_yu) in zip(range(int(eval_numg/config.eval.batch_size_gen)), ds_g_bpd, ds_u_bpd):
          #batch size train : eval = 5 : 1
          #same len
          score_model.eval()
          batch_xg = batch_xg.to(config.device)
          batch_xu = batch_xu.to(config.device)
          
          logging.info("uniform_dequantization")
          eval_batchg = datasetsgu.data_transform(config, batch_xg)
          eval_batchu = datasetsgu.data_transform(config, batch_xu)

          eval_batchg = scaler(eval_batchg)
          eval_batchu = scaler(eval_batchu)

          bpdgg = likelihood_fn(score_model, eval_batchg)[0]
          bpdgg = bpdgg.detach().cpu().numpy().reshape(-1)
          bpdggs.extend(bpdgg)
          
          bpduu = likelihood_fn(score_model, eval_batchu)[0]
          bpduu = bpduu.detach().cpu().numpy().reshape(-1)
          bpduus.extend(bpduu)

          logging.info('NLL not generated')
          logging.info("ckpt: %d, repeat: %d, batch: %d, mean bpd_un: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpduus))))
          bpd_round_idu = batch_id + len(ds_u_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_idu}_un.npz")
          with open(file_path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpduus)
            fout.write(io_buffer.getvalue())
          
          logging.info('NLL generated')
          logging.info("ckpt: %d, repeat: %d, batch: %d, mean bpd_gen: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpdggs))))
          bpd_round_idg = batch_id + len(ds_g_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          file_path = os.path.join(eval_dir, f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_idg}_gen.npz")
          with open(file_path, "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpdggs)
            fout.write(io_buffer.getvalue())
    
      
    #————————————————————————————————————————————————————————————————————————————————————————————————
    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:

      #===================================================#
      ######### generate samples and save samples #########
      #===================================================#
      need_samples = True
      if need_samples:
        print("sampling")
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size_sampling + 1
        for r in range(num_sampling_rounds):
        
          # Directory to save samples. Different for each host to avoid writing conflicts
          this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
          os.makedirs(this_sample_dir, exist_ok=True)

          samples, n = sampling_fn(score_model)
          # nrow = int(np.sqrt(sample.shape[0])) # 128   11
          # 1024 32
          nrow = config.eval.batch_size_sampling_nrow
          grid_image = make_grid(samples, nrow, padding=2)
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
          latents = evaluation.run_inception_distributed(samples, inception_model,
                                                        inceptionv3=inceptionv3)
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
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      stats = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz")) #glob.glob 返回匹配的文件路径列表
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
     
      data_pools = data_stats["pool_3"]

      # # Compute FID/KID/IS on all samples together.
      # Compute IS
      # if not inceptionv3:
      #     # Compute Inception Score
      #     from scipy.stats import entropy

      #     # Calculate probabilities
      #     preds = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True) #计算类别分布p(y|x) 
      #     #softmax 函数 

      #     # Calculate Inception Score
      #     kl_divs = preds * (np.log(preds) - np.log(np.mean(preds, axis=0, keepdims=True))) #p(y|x) * log p(y|x) - log p(y)
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

      # # Compute KID
      # dot_product = np.dot(data_pools, np.transpose(all_pools))
      # dot_product_exp = np.exp(np.sum(dot_product, axis=1, keepdims=True) / len(all_pools))
      # kid = np.mean(dot_product_exp)

      # Cleaning up
      # del dot_product, dot_product_exp

      # logging.info(
      #   "ckpt-%d --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (ckpt, inception_score, fid, kid))

      # Saving the report
      # report_file = os.path.join(eval_dir, f"report_{ckpt}.npz")
      # with open(report_file, "wb") as f:
      #     io_buffer = io.BytesIO()
      #     np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      #     f.write(io_buffer.getvalue())

      logging.info("ckpt-%d --- FID: %.6e" % (ckpt, fid))
      report_file = os.path.join(eval_dir, f"report_{ckpt}.npz")
      with open(report_file, "wb") as f:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, fid=fid)
          f.write(io_buffer.getvalue())
