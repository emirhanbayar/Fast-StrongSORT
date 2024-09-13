# Fast-StrongSORT

[![arXiv](https://img.shields.io/badge/arXiv-2409.06617-b31b1b.svg)](https://arxiv.org/abs/2409.06617)

## Data&Model Preparation

1. Download MOT17, MOT20 and DanceTrack datasets and organize them as follows:
(Note: Place the DanceTrack val set in the train set, and the test set in the test set.)

   ```
   data
   ├── MOT17
      ├── test
      └── train
   └── MOT20
      ├── test
      └── train
   └── dancetrack
       ├── test
       └── train
   ```

2. Download prepared detections and features for MOT17 and MOT20, which are provided by StrongSORT from [data](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_?usp=sharing) in Google drive (or [baidu disk](https://pan.baidu.com/s/1EtBbo-12xhjsqW5x-dYX8A?pwd=sort) with code "sort"). Place also weights for feature extractor (DukeMTMC_BoT-S50.pth) and place it under precomputed/.

Download the detections and features for DanceTrack, which are extracted using weights shared by OC-SORT from [data](https://drive.google.com/drive/folders/1k9mQWO3RJELN23Zs9jQmko5r8spt_zVP?usp=sharing) in Google drive. 

   ```
   precomputed
   ├── MOT17_test_YOLOX+BoT  # detections + features
   ├── MOT17_val_YOLOX+BoT  # detections + features
   ├── dancetrack_test_YOLOX+BoT # detections + features
   ├── dancetrack_val_YOLOX+BoT  # detections + features
   ├── MOT20_test_YOLOX+BoT  # detections + features
   ├── DukeMTMC_BoT-S50.pth
   ```

3. Set the paths of your dataset and other files in "opts.py", i.e., root_dataset dir_dets etc. The default paths are data/MOT17, data/MOT20, data/dancetrack, precomputed, etc. 

## Installation

A virtual environment is recommended to run the code. The code is tested on Python 3.8.19

```shell
pip install -r requirements.txt
```

## Tracking

Base StrongSORT
```shell
python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC
```

Fast-StrongSORT
```shell
python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --{offline} --{display} --ot {IoU_threshold} --aiou-threshold {ARS_threshold} --{decay_ema}
```

Where offline indicates using precomputed features, display indicates saving the tracking results into the path indicated in opts.py

--ot, --aiou-threshold, --decay_ema are the parameters that are introduced in Fast-StrongSORT, and explained in the paper.

## Example Commands for Best Configuration

```shell
python strong_sort.py MOT17 val --BoT --ECC --NSA --EMA --MC --woC --offline --display --ot 0.2 --aiou-threshold 0.6 --decay_ema 
python strong_sort.py MOT20 test --BoT --ECC --NSA --EMA --MC --woC --offline --display --ot 0.2 --aiou-threshold 0.6 --decay_ema 
python strong_sort.py dancetrack val --BoT --ECC --NSA --EMA --MC --woC --offline --display --ot 0.2 --aiou-threshold 0.6 --decay_ema
```

## Evaluation

Use TrackEval. If the scores are too low, you probably have not checked out the following issue: https://github.com/dyhBUPT/StrongSORT/issues/74
