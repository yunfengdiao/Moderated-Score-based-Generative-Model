
import run_lib
import run_lib3
import classcond
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import representation
import inpainting
import fid
current_path = os.getcwd()

os.sys.path.append(current_path)


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", '/home/jiang/home2/SDE/score_sde/configs/vp/ddpm/cifar10_continuous.py', "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", './resultsddpmc3_9505', "Work directory.")
flags.DEFINE_enum("mode", "eval", ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_enum("task", "square", ["normal", "no_u", "square", "norm","erase","y_cond","repre","inpaint","fid"], "Loss mode: train or eval")
# flags.DEFINE_enum("apply", "y_cond", ["y_cond", "no_u"], "Applyind task")
flags.DEFINE_enum("datasets_type", "D", ["D", "Dg", "Du"], "Loss mode: train or eval")
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.DEFINE_boolean('finetune', False, 'Whether to print verbose output')
#flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  if FLAGS.mode == "train":
    os.makedirs(FLAGS.workdir, exist_ok=True)
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'a') 
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    
    if FLAGS.task == "square" or FLAGS.task == "norm" or FLAGS.task == "erase":
      run_lib3.train(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.finetune)
    else:
      run_lib.train(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.finetune)


  elif FLAGS.mode == "eval":
    if FLAGS.task == "y_cond":
      eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
      os.makedirs(eval_dir, exist_ok=True)
      gfile_stream = open(os.path.join(eval_dir, 'cond_log.txt'), 'a')  
      
    elif FLAGS.task == "repre":
      eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
      os.makedirs(eval_dir, exist_ok=True)
      gfile_stream = open(os.path.join(eval_dir, 'repre_log.txt'), 'a')  
    elif FLAGS.task == "inpaint":
      eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
      os.makedirs(eval_dir, exist_ok=True)
      gfile_stream = open(os.path.join(eval_dir, 'inpaint_log.txt'), 'a')  
      
    else:
      eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
      os.makedirs(eval_dir, exist_ok=True)
      gfile_stream = open(os.path.join(eval_dir, 'eval_log.txt'), 'a') 


    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    logging.info(">" * 80)
    logging.info("eval folder =")
    logging.info(eval_dir)
    logging.info("<" * 80)

    # Run the training pipeline
    if FLAGS.task == "normal" or FLAGS.task == "no_u":
      # Run the evaluation pipeline
      run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder)
    elif FLAGS.task == "norm" or FLAGS.task == "square" or FLAGS.task == "erase":
      run_lib3.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder)
    elif FLAGS.task == "y_cond":
      classcond.classcond(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder)
    elif FLAGS.task == "repre":
      representation.repre(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder, FLAGS.datasets_type)
    elif FLAGS.task == "inpaint":
      inpainting.inpaint(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder, FLAGS.datasets_type)
    elif FLAGS.task == "fid":
      fid.fid(FLAGS.config, FLAGS.workdir, FLAGS.task, FLAGS.eval_folder, FLAGS.datasets_type)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)



