"""Creates an experiment (trains and monitors net, and saves relevant info)"""
import os
import sys
import shutil
#from object_mask_loss_2.train import train
from j_loss.train import train

ROOT_DIR = '/usr/people/kluther/Projects/metric_segmentation'

EXP_PATH = '/usr/people/kluther/seungmount/research/kluther/metric_segmentation/experiments'
DATA_DIR = '/usr/people/kluther/seungmount/research/kluther/metric_segmentation/data'



sys.path.append(os.path.join(ROOT_DIR, 'src'))

# GPUs
os.environ["CUDA_VISIBLE_DEVICES"]='0'

# Experiment parameters
EXP_NAME = 'test'
EXP_DIR =  os.path.join(EXP_PATH, EXP_NAME)
LOG_DIR = os.path.join(EXP_DIR, 'logs')
MODEL_DIR = os.path.join(EXP_DIR, 'models')
SAVE_DIR = os.path.join(EXP_DIR, 'saved')
SRC_DIR = os.path.join(EXP_DIR, 'src')

params = {'in_height': 572, # Network parameters
          'in_width': 572,
          'out_height': 388,
          'out_width': 388,
          'embed_dim': 32,
          'padding': 'SAME',
          'learning_rate': 0.0005,
          'n_sampled_objects': 1,
          'alpha': 1.0,
          'beta': 1.0,
          'exp_name': EXP_NAME,
          'exp_dir': EXP_DIR,
          'log_dir': LOG_DIR,
          'model_dir': MODEL_DIR,
          'save_dir': SAVE_DIR,
          'data_dir': DATA_DIR
}

# Create directories
if not os.path.exists(os.path.split(EXP_DIR)[0]):
  os.mkdir(os.path.split(EXP_DIR)[0])
if os.path.exists(EXP_DIR):
  msg = "Experiment with name '{}' already saved".format(EXP_NAME)
  raise FileExistsError(msg)
else:
  os.mkdir(EXP_DIR)

os.mkdir(LOG_DIR)
os.mkdir(MODEL_DIR)
os.mkdir(SAVE_DIR)
shutil.copytree(os.path.join(ROOT_DIR, 'src'), SRC_DIR)
open(os.path.join(EXP_DIR, 'notes.txt'), 'a').close()

# Run experiment
print("Running experiment: {}".format(EXP_NAME))
train(params)
