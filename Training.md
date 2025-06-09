## Training on RSDD-SAR and Custom Dataset
This guide shows how to use R-Sparse R-CNN with a custom dataset, using [SSDD](https://github.com/TianwenZhang0825/Official-SSDD) as an example. The same steps apply to other datasets.
Make sure the installation is complete and verified by following the steps in the [Installation Guide](./README.md) before proceeding.


### 1. Ground-truth Conversion
The SSDD dataset can be downloaded from [here](https://github.com/TianwenZhang0825/Official-SSDD).\
To train and evaluate using the SSDD dataset, convert the ground truth annotations to COCO format by running [`convert_SSDD_to_COCO_Detectron.py`](eval_json/convert_SSDD_to_COCO_Detectron.py).  
Set the `DATASET_phase` parameter to `"test"`, `"test_inshore"`, or `"test_offshore"` as needed. The script will generate the required JSON files upon success.

- [SSDD_test_COCO_OBB_Detectron.json](eval_json/SSDD_test_COCO_OBB_Detectron.json) 
- [SSDD_test_inshore_COCO_OBB_Detectron.json](eval_json/SSDD_test_inshore_COCO_OBB_Detectron.json) 
- [SSDD_test_offshore_COCO_OBB_Detectron.json](eval_json/SSDD_test_offshore_COCO_OBB_Detectron.json) 

### 2. Register the Dataset
Modify [builtin_meta.py](./detectron2/data/datasets/builtin_meta.py) to register your dataset.  
In that file, define the `categories` and `instances_meta`, and register them within the `_get_builtin_metadata` function.


### 3. Dataset Mapping
Depending on how your ground-truth boxes and labels are structured, you may need to implement a custom dataset mapper.  
For the SSDD dataset, an example implementation can be found in [dataset_mapper.py](projects/RSparseRCNN/rsparsercnn/dataset_mapper.py) and [ssdd_dataset.py](projects/RSparseRCNN/rsparsercnn/ssdd_dataset.py).

### 4. Configuration Files
Set the dataset and annotation `.json` paths (from Step 1) in [`config.py`](./projects/RSparseRCNN/rsparsercnn/config.py), and adjust model/training configs in [`Base-RSparseRCNN-OBB.yaml`](./projects/RSparseRCNN/configs/Base-RSparseRCNN.yaml) and [`rsparsercnn.res50.100pro.yaml`](./projects/RSparseRCNN/configs/rsparsercnn.res50.100pro.yaml).



### 5. Run the Training Script
```    
python projects/RSparseRCNN/train_net.py \
--num-gpus 1 --config-file projects/RSparseRCNN/configs/rsparsercnn.res50.100pro.yaml
```
Once training is finished, the trained model will be saved at `output/model_final.pth`.