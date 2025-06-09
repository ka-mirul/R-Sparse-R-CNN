# -*- coding: utf-8 -*-
# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_rsparsercnn_config(cfg):
    """
    Add config for SparseRCNN.
    """
    cfg.MODEL.RSparseRCNN = CN()
    cfg.MODEL.RSparseRCNN.NUM_CLASSES = 1
    cfg.MODEL.RSparseRCNN.NUM_PROPOSALS = 100

    # RCNN Head.
    cfg.MODEL.RSparseRCNN.NHEADS = 8 #self_attn
    cfg.MODEL.RSparseRCNN.DROPOUT = 0.0
    cfg.MODEL.RSparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.RSparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.RSparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.RSparseRCNN.NUM_CLS = 1
    cfg.MODEL.RSparseRCNN.NUM_REG = 3
    cfg.MODEL.RSparseRCNN.NUM_HEADS = 6 #6

    # Dynamic Conv.
    cfg.MODEL.RSparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.RSparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.RSparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.RSparseRCNN.IOU_WEIGHT = 2.0
    cfg.MODEL.RSparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.RSparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.RSparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.RSparseRCNN.USE_FOCAL = True
    cfg.MODEL.RSparseRCNN.ALPHA = 0.25
    cfg.MODEL.RSparseRCNN.GAMMA = 2.0
    cfg.MODEL.RSparseRCNN.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = "False"

    # Dataset
    cfg.CLASS_LABELS = ["ship"]
    cfg.TRAIN_DATASET_NAME = "RSDD_train" # also works for SSDD
    cfg.EVAL_DATASET_NAME = "RSDD_val" # also works for SSDD


    # Training Dataset

    cfg.DATASET_NAME = "SSDD" #for SSDD
    cfg.DATASET_MAIN_DIR = "/home/mikicil/xo23898/SHIP_DETECTION/DATASET/Official-SSDD-OPEN/RBox_SSDD/voc_style/"
    cfg.EVAL_IMAGES_DIR = "/home/mikicil/xo23898/SHIP_DETECTION/DATASET/Official-SSDD-OPEN/RBox_SSDD/voc_style/JPEGImages"
    cfg.EVAL_JSON_PATH = "eval_json/SSDD_test_COCO_OBB_Detectron.json"

    # cfg.DATASET_NAME = "RSDD" #for RSDD
    # cfg.DATASET_MAIN_DIR = r"/home/mikicil/xo23898/SHIP_DETECTION/DATASET/RSDD-SAR/"
    # cfg.EVAL_IMAGES_DIR = r"/home/mikicil/xo23898/SHIP_DETECTION/DATASET/RSDD-SAR/JPEGImages"
    # cfg.EVAL_JSON_PATH = r"eval_json/RSDD_test_COCO_OBB_Detectron.json"
    
