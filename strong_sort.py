"""
@Author: Du Yunhao
@Filename: strong_sort.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 20:14
@Discription: Run StrongSORT
"""
import warnings
from os.path import join
warnings.filterwarnings("ignore")
from opts import opt
from deep_sort_app import run

if __name__ == '__main__':

    total_extractions, total_frames, total_detections, total_ecc_time, total_prediciton_time, total_prematch_time, total_match_time, total_manag_time, total_embed_time = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i, seq in enumerate(opt.sequences, start=1):
        print('processing the {}th video {}...'.format(i, seq))
        path_save = join(opt.dir_save, seq + '.txt')
        extractions, detections, frames, ecc_time, prediciton_time, prematch_time, match_time, manag_time, embed_time, =  run(
            sequence_dir=join(opt.dir_dataset, seq),
            detection_file=join(opt.dir_dets, seq + '.npy'),
            output_file=path_save,
            min_confidence=opt.min_confidence,
            nms_max_overlap=opt.nms_max_overlap,
            min_detection_height=opt.min_detection_height,
            max_cosine_distance=opt.max_cosine_distance,
            nn_budget=opt.nn_budget,
            display=opt.display,
        )

        total_extractions += extractions
        total_frames += frames
        total_detections += detections
        total_ecc_time += ecc_time
        total_prediciton_time += prediciton_time
        total_prematch_time += prematch_time
        total_match_time += match_time
        total_manag_time += manag_time
        total_embed_time += embed_time
        # print("Extracted %d features in total" % extractions)
        # print("Extraction per Frame (EPF): %.2f" % (extractions / frames))
        # print("Detections per Frame (DPF): %.2f" % (detections / frames))
        # print("Percentage of Detections Extracted (PDE): %.2f" % (extractions / detections * 100))

    # print('All videos are processed...')
    print("************************* OT: ", opt.occlusion_threshold, " AIoU: ", opt.aiou_threshold, "Decay: ", opt.decay_ema, "PreMatch: ", opt.prematch, "*************************")
    print("Extracted %d features in total" % total_extractions)
    print("Extraction per Frame (EPF): %.2f" % (total_extractions / total_frames))
    print("Detections per Frame (DPF): %.2f" % (total_detections / total_frames))
    print("Percentage of Detections Extracted (PDE): %.2f" % (total_extractions / total_detections * 100))
    print("ECC Time: %.2f" % total_ecc_time)
    print("Prediction Time: %.2f" % total_prediciton_time)
    print("Prematch Time: %.2f" % total_prematch_time)
    print("Match Time: %.2f" % total_match_time)
    print("Embed Time: %.2f" % total_embed_time)
    print("Manag Time: %.2f" % total_manag_time)
    print("FPS: %.2f" % (total_frames / (total_ecc_time + total_prediciton_time + total_prematch_time + total_match_time + total_embed_time + total_manag_time)))
    print("Total Frames: %.2f" % (total_frames))
    print("*******************************************************************************************************************")
    

