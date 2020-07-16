from lib.utils.misc import detect_os

MOT_info = {
    "readme": "The sequences in MOT16 and MOT17 are the same, while the sequences in 2DMOT2015 are "
              "not all the same with those in MOT17. To handle this, we filter 2DMOT2015 dataset, "
              "i.e. those sequences that are contained in MOT17 will not be included in 2DMOT2015",


    'year': ['MOT15', 'MOT16', 'MOT17', 'MOT20'],

    'base_path': {
        'MOT15': {
            'im': '/home/liuqk/Dataset/MOT/2DMOT2015',
            'label':'/home/liuqk/Dataset/MOT/2DMOT2015',
        },
        'MOT16': {
            'im': '/home/liuqk/Dataset/MOT/MOT17Det',
            'label':'/home/liuqk/Dataset/MOT/MOT16Labels',
        },
        'MOT17': {
            'im': '/home/liuqk/Dataset/MOT/MOT17Det',
            'label': '/home/liuqk/Dataset/MOT/MOT17Labels',
        },
        'MOT20': {
            'im': '/home/liuqk/Dataset/MOT/MOT20',
            'label': '/home/liuqk/Dataset/MOT/MOT20',
        },
        'MOT16-det-dpm-raw': '/home/liuqk/Dataset/MOT/MOT16-det-dpm-raw/',
    },
    'sequences': {
        'MOT15': {
            'train': ['ETH-Bahnhof', 'ETH-Sunnyday', 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
                      'TUD-Stadtmitte'],
            'test': ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
                     'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1'],
            'val': []
        },

        'MOT16': {
            'train': ['MOT16-04', 'MOT16-11', 'MOT16-05', 'MOT16-13', 'MOT16-02'], #, 'MOT16-10', 'MOT16-09'],
            'test': ['MOT16-12', 'MOT16-03', 'MOT16-01', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14'],
            'val': ['MOT16-09', 'MOT16-10']
        },
        'MOT17': {
            'train': ['MOT17-04', 'MOT17-11', 'MOT17-05', 'MOT17-13', 'MOT17-02'],
            'test': ['MOT17-03', 'MOT17-01', 'MOT17-06', 'MOT17-07', 'MOT17-08', 'MOT17-12', 'MOT17-14'],
            'val': ['MOT17-10', 'MOT17-09']
        },
        'MOT20':{
            'train':['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
            'test': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'],
            'val': [],
        },
    },
    'detectors': {
        'MOT15': {
            'train': [''],
            'val': [''],
            'test': ['']
        },
        'MOT16': {
            'train': [''],
            'val': [''],
            'test': ['']
        },
        'MOT17': {
            'train': ['DPM'],
            'val': ['DPM'], #['SDP', 'FRCNN', 'DPM'],
            'test': ['DPM', 'SDP', 'FRCNN']
        },
        'MOT20': {
            'train': [''],
            'val': [''],
            'test': ['']
        },
    },
}

def get_mot_info():
    mot_info = MOT_info.copy()

    # modify the path based on the OS
    operate_system = detect_os()
    if operate_system == 'MAC_OS_X':
        base_path = {
            'MOT15': {
                'im': '/Users/Qiankun/Learning/Dataset/MOT/2DMOT2015',
                'label': '/Users/Qiankun/Learning/Dataset/MOT/2DMOT2015',
            },
            'MOT16': {
                'im': '/Users/Qiankun/Learning/Dataset/MOT/MOT17Det',
                'label': '/Users/Qiankun/Learning/Dataset/MOT/MOT16Labels',
            },
            'MOT17': {
                'im': '/Users/Qiankun/Learning/Dataset/MOT/MOT17Det',
                'label': '/Users/Qiankun/Learning/Dataset/MOT/MOT17Labels',
            },
            'MOT20': {
                'im': '/Users/Qiankun/Learning/Dataset/MOT/MOT20',
                'label': '/Users/Qiankun/Learning/Dataset/MOT/MOT20',
            },
            'MOT16-det-dpm-raw': '/home/liuqk/Dataset/MOT/MOT16-det-dpm-raw/',
        }

    elif operate_system == 'WINDOWS':
        base_path = {
            'MOT15': {
                'im': 'F:\Datasets\MOT2DMOT2015',
                'label': 'F:\Datasets\MOT2DMOT2015',
            },
            'MOT16': {
                'im': 'F:\Datasets\MOT\MOT16',
                'label': 'F:\Datasets\MOT\MOT16',
            },
            'MOT17': {
                'im': 'F:\Datasets\MOT\MOT17',
                'label': 'F:\Datasets\MOT\MOT17',
            },
            'MOT20': {
                'im': 'F:\Datasets\MOT\MOT20',
                'label': 'F:\Datasets\MOT\MOT20',
            },
            'MOT16-det-dpm-raw': '/home/liuqk/Dataset/MOT/MOT16-det-dpm-raw/',
        }
    elif operate_system == 'LINUX':
        base_path = {
            'MOT15': {
                'im': '/home/liuqk/Dataset/MOT/2DMOT2015',
                'label': '/home/liuqk/Dataset/MOT/2DMOT2015',
            },
            'MOT16': {
                # 'im': '/home/liuqk/Dataset/MOT/MOT17Det',
                # 'label': '/home/liuqk/Dataset/MOT/MOT16Labels',
                'im': '/home/liuqk/Dataset/MOT/MOT16',
                'label':'/home/liuqk/Dataset/MOT/MOT16',
            },
            'MOT17': {
                # 'im': '/home/liuqk/Dataset/MOT/MOT17Det',
                # 'label': '/home/liuqk/Dataset/MOT/MOT17Labels',
                'im': '/home/liuqk/Dataset/MOT/MOT17',
                'label':'/home/liuqk/Dataset/MOT/MOT17',
            },
            'MOT20': {
                'im': '/home/liuqk/Dataset/MOT/MOT20',
                'label': '/home/liuqk/Dataset/MOT/MOT20',
            },
            'MOT16-det-dpm-raw': '/home/liuqk/Dataset/MOT/MOT16-det-dpm-raw/',
        }
    else:
        raise NotImplementedError('Unkonwn operating system {}'.format(operate_system))

    mot_info['base_path'] = base_path

    return mot_info
