# Fast-StrongSORT

[![arXiv](https://img.shields.io/badge/arXiv-2409.06617-b31b1b.svg)](https://arxiv.org/abs/2409.06617)

## Performance Metrics
### FPS Speed-up Table

| Configuration                     | FPS (osnet_x1 GTX1650) | FPS (mobilenetv2_x1_0 GTX1650) | FPS (osnet_x0_25 CPU) | FPS (osnet_x1 T4) | FPS (ResNet50 TX2) |
|-----------------------------------|------------------------|--------------------------------|------------------------|-------------------|---------------------|
| StrongSORT                        | 4.85                   | 5.09                           | 1.33                   | 5.80              | 0.96                |
| Fast-StrongSORT (iou_threshold=0.0)| 5.23 (+8%)             | 5.33 (+5%)                     | 1.68 (+26%)            | 6.67 (+15%)       | 1.17 (+22%)         |
| Fast-StrongSORT (iou_threshold=0.1)| 5.59 (+15%)            | 5.57 (+9%)                     | 2.02 (+52%)            | 7.22 (+24%)       | 1.40 (+46%)         |
| Fast-StrongSORT (iou_threshold=0.2)| 5.97 (+23%)            | 5.86 (+15%)                    | 2.48 (+86%)            | 7.71 (+33%)       | 1.73 (+80%)         |
| Fast-StrongSORT (iou_threshold=0.3)| 6.42 (+32%)            | 6.11 (+20%)                    | 2.94 (+121%)           | 8.32 (+43%)       | 2.07 (+116%)        |
| Fast-StrongSORT (iou_threshold=0.4)| 6.84 (+41%)            | 6.34 (+25%)                    | 3.64 (+174%)           | 8.80 (+52%)       | 2.52 (+163%)        |
| Fast-StrongSORT (iou_threshold=0.5)| 7.23 (+49%)            | 6.57 (+29%)                    | 4.51 (+239%)           | 9.22 (+59%)       | 3.21 (+234%)        |

### MOT17 Val Performance Metrics

| Tracker     | PDE* ↓       | FET** (s) ↓  | FPS ↑        | HOTA ↑       | AssA ↑       | IDF1 ↑       |
|-------------|--------------|--------------|--------------|--------------|--------------|--------------|
| StrongSORT  | 100.0        | 2457         | 0.96         | 69.54        | 73.29        | 82.17        |
| FSS 0.0     | 76.38 (-24%) | 1946 (-21%)  | 1.17 (+22%)  | 69.36 (-0.26%)| 72.92 (-0.50%)| 81.88 (-0.35%)|
| FSS 0.1     | 58.67 (-41%) | 1598 (-35%)  | 1.40 (+46%)  | 69.39 (-0.22%)| 72.93 (-0.49%)| 81.81 (-0.44%)|
| FSS 0.2     | 43.93 (-56%) | 1241 (-50%)  | 1.73 (+80%)  | 69.61 (+0.1%) | 73.44 (+0.21%)| 82.56 (+0.47%)|
| FSS 0.3     | 33.40 (-67%) | 995 (-60%)   | 2.07 (+116%) | 69.09 (-0.65%)| 72.50 (-1.07%)| 81.97 (-0.24%)|
| FSS 0.4     | 23.46 (-77%) | 772 (-69%)   | 2.52 (+163%) | 68.31 (-1.77%)| 70.87 (-3.30%)| 80.91 (-1.53%)|
| FSS 0.5     | 16.14 (-84%) | 547 (-78%)   | 3.21 (+234%) | 68.86 (-0.98%)| 71.88 (-1.92%)| 81.21 (-1.17%)|

*PDE is the abbreviation of Percentage of Detections with Extraction.
**FET is the abbreviation of Total Feature Extraction Time.

### DanceTrack Val Performance Metrics

| Tracker     | PDE* ↓       | FET** (s) ↓  | FPS ↑        | HOTA ↑       | AssA ↑       | IDF1 ↑       |
|-------------|--------------|--------------|--------------|--------------|--------------|--------------|
| StrongSORT  | 100.0        | 14054        | 1.56         | 56.61        | 41.12        | 55.91        |
| FSS 0.0     | 85.34 (-15%) | 11972 (-15%) | 1.79 (+15%)  | 57.07 (+0.81%)| 41.78 (+1.61%)| 56.35 (+0.77%)|
| FSS 0.1     | 73.40 (-27%) | 10518 (-25%) | 2.00 (+28%)  | 56.55 (-0.11%)| 41.01 (-0.27%)| 55.90 (-0.03%)|
| FSS 0.2     | 58.64 (-41%) | 8548 (-39%)  | 2.38 (+53%)  | 56.80 (+0.34%)| 41.36 (+0.60%)| 56.19 (+0.49%)|
| FSS 0.3     | 43.02 (-57%) | 6434 (-54%)  | 2.97 (+90%)  | 55.46 (-2.03%)| 39.38 (-4.22%)| 54.06 (-3.32%)|
| FSS 0.4     | 27.97 (-72%) | 4264 (-70%)  | 4.02 (+158%) | 53.92 (-4.75%)| 37.29 (-9.32%)| 52.89 (-5.40%)|
| FSS 0.5     | 16.28 (-84%) | 2540 (-82%)  | 5.64 (+262%) | 51.55 (-8.94%)| 34.17 (-16.90%)| 50.92 (-8.93%)|

*PDE is the abbreviation of Percentage of Detections with Extraction.
**FET is the abbreviation of Total Feature Extraction Time.



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
