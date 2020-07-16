import matplotlib.pyplot as plt
import os
import cv2
from lib.utils.visualization import plot_tracking
import numpy as np
import glob


def show_tracks(file_list, im_root):
    row = 1
    col = max(1, len(file_list))

    box_list = []
    for f in file_list:
        box = np.loadtxt(fname=f, delimiter=',')
        box_list.append(box)

    im_name = glob.glob(os.path.join(im_root, '*.jpg'))
    frame_id = list(range(1, len(im_name)+1))

    for fid in frame_id:
        # if fid < 15:
        #     continue
        im_name = os.path.join(im_root, str(fid).zfill(6)+'.jpg')
        im = cv2.imread(im_name)
        for idx in range(len(box_list)):
            box = box_list[idx]
            frame_box = box[box[:, 0] == fid]
            tlwhs = frame_box[:, 2:6]
            ids = frame_box[:, 1]
            im_tmp = plot_tracking(image=im.copy(), tlwhs=tlwhs, obj_ids=ids, frame_id=fid)
            im_tmp = im_tmp[:, :, ::-1]

            # if fid in [27, 29, 49]:
            #     plt.imsave('./shot_images/frame_{}_{}.jpg'.format(fid, idx+1), im_tmp)

            if idx == 0:
                plt.clf()

            plt.subplot(row, col, idx+1)
            plt.imshow(im_tmp)
            plt.axis('off')

        plt.pause(0.001)
        # if fid == 50:
        #     break
        a = 1


if __name__ == '__main__':

    box1 = '/home/liuqk/Program/pycharm/MOT-graph/track_results/MOTDT/nodet_nocls_nokf_noflow_noiou_nogate_notrack_norefind/public/MOT20/test/graph_match_nei/GraphSimilarity_v5_2/0.85/22/MOT20-08.txt'

    mot_root = '/home/liuqk/Dataset/MOT/'

    if 'MOT16' in box1:
        year = 'MOT16'
    elif 'MOT17' in box1:
        year = 'MOT17'
    elif 'MOT20' in box1:
        year = 'MOT20'
    else:
        year = '2DMOT2015'

    if 'train' in box1 or 'val' in box1:
        phase = 'train'
    else:
        phase = 'test'

    base_name = os.path.basename(box1)
    seq_name = base_name.split('.')[0]
    im_root = os.path.join(mot_root, year, phase, seq_name, 'img1')

    show_tracks(file_list=[box1], im_root=im_root)

