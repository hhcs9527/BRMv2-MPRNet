import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRBRRNet import MPRBRRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from config import Config 

opt = Config('training_lite.yml')
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION
result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

if not os.path.exists(result_dir): os.mkdir(result_dir)
if not os.path.exists(model_dir): os.mkdir(model_dir)
