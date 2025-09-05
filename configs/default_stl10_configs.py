import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 200 #128 
  config.training.batch_size_gen = 400
  config.training.batch_size_un = 100
  training.n_iters = 1300001      
  training.snapshot_freq = 20000      
  training.log_freq = 40          
  training.eval_freq = 40       
  training.snapshot_freq_for_preemption = 10000   
  ## produce samples at each snapshot.  
  training.snapshot_sampling = True  
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt =  35 
  evaluate.end_ckpt = 35  
  evaluate.batch_size = 500
  
  #1024
  evaluate.batch_size_gen = 200 #1024
  evaluate.batch_size_un = 200 #1024
  evaluate.batch_size_sampling = 1000 #1024
  evaluate.batch_size_sampling_nrow = 25 #1024
  evaluate.enable_sampling = True  #False ********｜IS/FID/KID｜***************
  evaluate.num_samples = 50000
  evaluate.enable_loss = False        #**********｜eval loss｜*************
  evaluate.enable_bpd = False #False  #************｜likelihood｜***********
  evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'STL10'
  data.data_path = './data/'
  data.image_size = 64
  data.random_flip = True
  data.centered = True #False
  data.uniform_dequantization = False #train F /eval NLL True
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
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