
import os
import cv2
import sys
import glob
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from torchvision import transforms
from os.path import join, exists, split

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

def get_transform(size=(256, 128)):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # norm,
    ])
    return transform

class Embedder():
    def __init__(self):
        # disable print
        sys.stdout = open(os.devnull, 'w')

        torch.cuda.empty_cache()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        args = default_argument_parser().parse_args()
        args.eval_only = True
        args.config_file = 'configs/DukeMTMC/bagtricks_S50.yml'
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.freeze()
        default_setup(cfg, args)
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = 'precomputed/DukeMTMC_BoT-S50.pth'
        self.thres_score = 0.6
        self.model = DefaultTrainer.build_model(cfg)
        self.model.eval()
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.transform = get_transform((256, 128))

        # enable print
        sys.stdout = sys.__stdout__

    def inference(self, dets, frame):
        batch = [frame.crop((b[0], b[1], b[0] + b[2], b[1] + b[3])) for b in dets]
        transformed_batch = [self.transform(patch) * 255. for patch in batch]
        if transformed_batch:
            transformed_batch = torch.stack(transformed_batch, dim=0).cuda()
            with torch.no_grad():
                outputs = self.model(transformed_batch).detach().cpu().numpy()
        else:
            outputs = np.array([None])

        return outputs



        