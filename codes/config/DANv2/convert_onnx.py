import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import OrderedDict
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
from torch.autograd import Variable

import torch.onnx
import torchvision
import torch



class class_parse(object):
    def __init__(self):
        self.opt="/root/hmrishav/DAN/codes/config/DANv2/options/setting1/test/test_setting1_x2.yml"


args = class_parse()
opt = option.parse(args.opt, is_train=False)

opt = option.dict_to_nonedict(opt)

model = create_model(opt)
# model = torchvision.models.mobilenet_v2(pretrained=True)#model.ret_model()._modules['module'].cpu()
model = model.ret_model()._modules['module'].cpu()
dummy_input = Variable(torch.randn(1, 3, 256, 256)).cpu()
state_dict = torch.load('/root/hmrishav/DAN/experiments/DANv2/DANx2_setting1/models/latest_G.pth')
model.load_state_dict(state_dict)
torch.onnx.export(model, dummy_input, "out_mod.onnx",verbose=True)