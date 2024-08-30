import os

datasets = ["dancetrack"]
dataset_types = ["test"]
# occlusion_thresholds 0.0:1.0:0.05 rounded to second decimal
# occlusion_thresholds = np.round(np.arange(0.0, 1.05, 0.05), 2)
occlusion_thresholds = [0.2]
aiou_thresholds = [0.6]
ema_decay = [True]
prematches = [False]

# occlusion_thresholds = [0.0]

for dataset in datasets:
    for dataset_type in dataset_types:
        for th in occlusion_thresholds:
            for aiou in aiou_thresholds:
                for decay in ema_decay:
                    for prematch in prematches:
                        if decay:
                            if prematch:
                                if dataset == "dancetrack" or dataset == "MOT20":
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --decay_ema --aiou-threshold {aiou} --display --prematch")
                                else:
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --decay_ema --aiou-threshold {aiou} --display --prematch")
                            else:
                                if dataset == "dancetrack" or dataset == "MOT20":
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --decay_ema --aiou-threshold {aiou} --display")
                                else:
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --decay_ema --aiou-threshold {aiou} --display")
                        else:
                            if prematch:
                                if dataset == "dancetrack" or dataset == "MOT20":
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --aiou-threshold {aiou} --display --prematch")
                                else:
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --aiou-threshold {aiou} --display --prematch")
                            else:
                                if dataset == "dancetrack" or dataset == "MOT20":
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --aiou-threshold {aiou} --display")
                                else:
                                    os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --aiou-threshold {aiou} --display")

            # # python strong_sort.py MOT17 val --BoT --ECC --NSA --EMA --MC --woC -pt 0.0
            # if dataset == "dancetrack" or dataset == "MOT20":
            #     os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th}")
            # else:
            #     os.system(f"python strong_sort.py {dataset} {dataset_type} --BoT --ECC --NSA --EMA --MC --woC --ot {th} --decay_ema --aiou-threshold 0.7")
