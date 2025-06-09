# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
RSparseRCNN Training Script..

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
from typing import Any, Dict, List, Set
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import RotatedCOCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from rsparsercnn import RSparseRCNNDatasetMapper, add_rsparsercnn_config
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata



class Trainer(DefaultTrainer): # from detectron2/engine/defaults.py
#     """
#     Extension of the Trainer class adapted to SparseRCNN-OBB.
#     """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = RSparseRCNNDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_rsparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    test_dataset_name = "RSDD_val"
    cfg = setup(args)
    MODEL_PATH = r"output/model_final.pth"


    if cfg.DATASET_NAME == 'SSDD': 
        print('#'*200)
        print('Testing SSDD Offshore dataset')
        print('#'*200)

        # MODEL_PATH = r"saved_models/model_SSDD.pth"
        # REGISTER THE IMAGES & GROUND TRUTHS
        JSON_PATH = r"eval_json/SSDD_test_offshore_COCO_OBB_Detectron.json"
        data_dir = r"/home/mikicil/xo23898/SHIP_DETECTION/DATASET/Official-SSDD-OPEN/RBox_SSDD/voc_style/"
        register_coco_instances(test_dataset_name, _get_builtin_metadata("rsdd"), 
                            JSON_PATH,
                            os.path.join(data_dir, "JPEGImages"))
        
    elif cfg.DATASET_NAME == 'RSDD': 
        print('#'*200)
        print('Testing RSDD Offshore dataset')
        print('#'*200)
        # MODEL_PATH = r"saved_models/model_RSDD.pth"

        # REGISTER THE IMAGES & GROUND TRUTHS
        JSON_PATH = r"eval_json/RSDD_test_offshore_COCO_OBB_Detectron.json"
        data_dir = r"/home/mikicil/xo23898/SHIP_DETECTION/DATASET/RSDD-SAR/"
        register_coco_instances(test_dataset_name, _get_builtin_metadata("rsdd"), 
                            JSON_PATH, 
                            os.path.join(data_dir, "JPEGImages"))

    model = Trainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(MODEL_PATH)
    Trainer.test(cfg, model)


if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
