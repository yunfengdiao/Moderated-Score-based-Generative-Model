
import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 150  
  config.training.batch_size_gen = 150
  config.training.batch_size_un = 150
  training.n_iters = 1300001      
  training.snapshot_freq = 2000      
  training.log_freq = 40           
  training.eval_freq = 40         
  training.snapshot_freq_for_preemption = 1000   
  ## produce samples at each snapshot.  
  training.snapshot_sampling = True  
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # config.training = training = ml_collections.ConfigDict()
  # config.training.batch_size = 128
  # training.n_iters = 1300001
  # training.snapshot_freq = 50000
  # training.log_freq = 50
  # training.eval_freq = 100
  # ## store additional checkpoints for preemption in cloud computing environments
  # training.snapshot_freq_for_preemption = 10000
  # ## produce samples at each snapshot.
  # training.snapshot_sampling = True
  # training.likelihood_weighting = False
  # training.continuous = True
  # training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt =  45678 #
  evaluate.end_ckpt = 45679  #
  evaluate.batch_size = 200 #1024
  evaluate.batch_size_gen = 200 #1024
  evaluate.batch_size_un = 200 #1024
  evaluate.batch_size_sampling = 200 #1024
  evaluate.batch_size_sampling_nrow = 25 #1024
  evaluate.enable_sampling = True  #False ********｜IS/FID/KID｜***************
  evaluate.num_samples = 50000
  evaluate.enable_loss = False        #**********｜eval loss｜*************
  evaluate.enable_bpd = False #False  #************｜likelihood｜***********
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CELEBA'
  data.data_path = '/home/jiang/home2/data'
  data.image_size = 64
  data.random_flip = False
  data.uniform_dequantization = False
  data.centered = True
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
# import ml_collections
# import torch


# def get_default_configs():
#   config = ml_collections.ConfigDict()
#   # training
#   config.training = training = ml_collections.ConfigDict()
#   config.training.batch_size = 200 #128 
#   config.training.batch_size_gen = 128
#   config.training.batch_size_un = 128
#   training.n_iters = 1300001      #总迭代step 数  1300001 
#   training.snapshot_freq = 2000      #保存模型采样    50000
#   training.log_freq = 40           #训练日志输出频率  50
#   training.eval_freq = 40         #多少 step eval  100
#   training.snapshot_freq_for_preemption = 1000   #保存ckpt to resume training every 10000
#   ## produce samples at each snapshot.  
#   training.snapshot_sampling = True   #是否生成样本并保存
#   training.likelihood_weighting = False
#   training.continuous = True
#   training.reduce_mean = False

#   # config.training = training = ml_collections.ConfigDict()
#   # config.training.batch_size = 128
#   # training.n_iters = 1300001
#   # training.snapshot_freq = 50000
#   # training.log_freq = 50
#   # training.eval_freq = 100
#   # ## store additional checkpoints for preemption in cloud computing environments
#   # training.snapshot_freq_for_preemption = 10000
#   # ## produce samples at each snapshot.
#   # training.snapshot_sampling = True
#   # training.likelihood_weighting = False
#   # training.continuous = True
#   # training.reduce_mean = False

#   # sampling
#   config.sampling = sampling = ml_collections.ConfigDict()
#   sampling.n_steps_each = 1
#   sampling.noise_removal = True
#   sampling.probability_flow = False
#   sampling.snr = 0.16

#   # evaluation
#   config.eval = evaluate = ml_collections.ConfigDict()
#   evaluate.begin_ckpt =  45678 #kaishi
#   evaluate.end_ckpt = 45678  #
#   evaluate.batch_size = 200 #1024
#   evaluate.batch_size_gen = 200 #1024
#   evaluate.batch_size_un = 200 #1024
#   evaluate.batch_size_sampling = 400 #1024
#   evaluate.batch_size_sampling_nrow = 20 #1024
#   evaluate.enable_sampling = True  #False ********｜IS/FID/KID｜***************
#   evaluate.num_samples = 50000
#   evaluate.enable_loss = False        #**********｜eval loss｜*************
#   evaluate.enable_bpd = False #False  #************｜likelihood｜***********
#   evaluate.bpd_dataset = 'test'

#   # data
#   config.data = data = ml_collections.ConfigDict()
#   data.dataset = 'CELEBA'
#   data.data_path = '/home/jiang/home2/data'
#   data.image_size = 64
#   data.random_flip = False
#   data.uniform_dequantization = False
#   data.centered = True
#   data.num_channels = 3

#   # model
#   config.model = model = ml_collections.ConfigDict()
#   model.sigma_max = 90.
#   model.sigma_min = 0.01
#   model.num_scales = 1000
#   model.beta_min = 0.1
#   model.beta_max = 20.
#   model.dropout = 0.1
#   model.embedding_type = 'fourier'

#   # optimization
#   config.optim = optim = ml_collections.ConfigDict()
#   optim.weight_decay = 0
#   optim.optimizer = 'Adam'
#   optim.lr = 2e-4
#   optim.beta1 = 0.9
#   optim.eps = 1e-8
#   optim.warmup = 5000
#   optim.grad_clip = 1.

#   config.seed = 42
#   config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#   return config