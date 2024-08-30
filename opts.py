"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join

data = {
    'MOT17': {
        'val':[
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN'
        ],
        'test':[
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN'
        ]
    },
    'MOT20': {
        'val':[
            'MOT20-01',
            'MOT20-02',
            'MOT20-03',
            'MOT20-05'
        ],
        'test':[
            'MOT20-04',
            'MOT20-06',
            'MOT20-07',
            'MOT20-08'
        ]
    },
    'dancetrack': {
        'val':[
               'dancetrack0004',
               'dancetrack0005',
               'dancetrack0007',
               'dancetrack0010',
               'dancetrack0014',
               'dancetrack0018',
               'dancetrack0019',
               'dancetrack0025',
               'dancetrack0026',
               'dancetrack0030',
               'dancetrack0034',
               'dancetrack0035',
               'dancetrack0041',
               'dancetrack0043',
               'dancetrack0047',
               'dancetrack0058',
               'dancetrack0063',
               'dancetrack0065',
               'dancetrack0073',
               'dancetrack0077',
               'dancetrack0079',
               'dancetrack0081',
               'dancetrack0090',
               'dancetrack0094',
               'dancetrack0097',
        ],
        'test':[
                'dancetrack0003',
                'dancetrack0009',
                'dancetrack0011',
                'dancetrack0013',
                'dancetrack0017',
                'dancetrack0021',
                'dancetrack0022',
                'dancetrack0028',
                'dancetrack0031',
                'dancetrack0036',
                'dancetrack0038',
                'dancetrack0040',
                'dancetrack0042',
                'dancetrack0046',
                'dancetrack0048',
                'dancetrack0050',
                'dancetrack0054',
                'dancetrack0056',
                'dancetrack0059',
                'dancetrack0060',
                'dancetrack0064',
                'dancetrack0067',
                'dancetrack0070',
                'dancetrack0071',
                'dancetrack0076',
                'dancetrack0078',
                'dancetrack0084',
                'dancetrack0085',
                'dancetrack0088',
                'dancetrack0089',
                'dancetrack0091',
                'dancetrack0092',
                'dancetrack0093',
                'dancetrack0095',
                'dancetrack0100'
        ]
    }
}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            'dataset',
            type=str,
            help='MOT17 or MOT20',
        )
        self.parser.add_argument(
            'mode',
            type=str,
            help='val or test',
        )
        self.parser.add_argument(
            '--BoT',
            action='store_true',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLink',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='./data/'
        )
        self.parser.add_argument(
            '--path_AFLink',
            default='./precomputed/AFLink_epoch20.pth'
        )
        self.parser.add_argument(
            '--dir_save',
            default='./precomputed/tmp'
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98
        )
        self.parser.add_argument(
            '--occlusion_threshold',
            '--ot',
            type=float,
            default=0.5
        )
        self.parser.add_argument(
            '--display',
            action='store_true',
            help='Display the tracking results'
        )
        self.parser.add_argument(
            '--offline',
            action='store_true',
            help='Offline tracking'
        )
        self.parser.add_argument(
            '--decay_ema',
            action='store_true',
            help='Decay EMA alpha'
        )
        self.parser.add_argument(
            '--aiou-threshold',
            type=float,
            default=0.7
        )
        self.parser.add_argument(
            '--cheat-update',
            action='store_true',
            help='Cheat update'
        )
        self.parser.add_argument(
            '--record',
            action='store_true',
            help='Record the tracking results'
        )

        self.parser.add_argument(
            '--prematch',
            action='store_true',
            help='Prematch'
        )


    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.6
        opt.nms_max_overlap = 1.0
        opt.min_detection_height = 0
        if opt.BoT:
            opt.max_cosine_distance = 0.40
            opt.dir_dets = './precomputed/{}_{}_YOLOX+BoT'.format(opt.dataset, opt.mode)
        else:
            opt.max_cosine_distance = 0.3
            opt.dir_dets = './precomputed/{}_{}_YOLOX+simpleCNN'.format(opt.dataset, opt.mode)
        if opt.MC:
            opt.max_cosine_distance += 0.05
        if opt.EMA:
            opt.nn_budget = 1
        else:
            opt.nn_budget = 100
        opt.sequences = data[opt.dataset][opt.mode]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            'train' if opt.mode == 'val' else 'test'
        )
        return opt

opt = opts().parse()