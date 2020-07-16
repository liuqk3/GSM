import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import motmetrics as mm
from lib.tracking.tracker import OnlineTracker

from lib.datasets.mot_seq import get_seq_loader
from lib.datasets.mot_info import get_mot_info
from lib.utils import visualization as vis
from lib.utils.timer import Timer
from lib.utils.evaluation import Evaluator
from lib.utils.io import images2video
import matplotlib.pyplot as plt
from collections import deque
import torch
import numpy as np

MOT_info = get_mot_info()


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    print('save results to {}'.format(filename))


def eval_seq(dataloader, data_type, result_filename, im_track_save_dir=None, plot_type='tracks',
             anchor_id_to_show=0, tracker_args={}):
    if im_track_save_dir is not None:
        if not os.path.exists(im_track_save_dir):
            os.makedirs(im_track_save_dir)

    tracker_args['detector_name'] = dataloader.dataset.detector
    tracker = OnlineTracker(**tracker_args)
    timer = Timer()
    results = []
    history_im = deque([], maxlen=4)
    for frame_id, batch_data in enumerate(dataloader):

        if (frame_id + 1) % 10 == 0:
            print('Processing {}: frame {}/{} ({:.2f} fps)'.format(dataloader.dataset.seq_name, frame_id+1,
                                                                   len(dataloader), 1./max(1e-5, timer.average_time)))
            #print('score_time: {}, construct_time: {}'.format(tracker.asso_model.score_time, tracker.asso_model.construct_time))
        frame = batch_data['im']
        frame_tensor = frame.astype(np.float32)
        frame_tensor = torch.Tensor(frame_tensor).cuda()  # [h, w, 3]

        tlwhs = batch_data['tlwhs']
        flow = batch_data['flow']
        scores = batch_data['scores']
        # run tracking
        timer.tic()
        online_targets = tracker.update(image=frame_tensor, tlwhs=tlwhs,
                                        flow=flow, det_scores=scores)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh())
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        if plot_type == 'tracks':
            online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id+1,
                                          fps=1. / timer.average_time)
        elif plot_type == 'neighbors':
            track = None
            if anchor_id_to_show == 0 and len(online_targets) > 0:
                track = online_targets[0]
            else:
                for t in online_targets:
                    if t.track_id == anchor_id_to_show:
                        track = t
                        break
            if track is None:
                print('Anchor track with ID {} if not found, so no neighbors will be ploted'.format(anchor_id_to_show))
                online_im = frame
            else:
                online_im = vis.plot_neighbors(image=frame, tlwh_anchor=track.tlwh(), obj_id_anchor=track.track_id,
                                               tlwh_nei=track.tlwh_nei(), weight_nei=track.weight_nei(),
                                               frame_id=frame_id+1)
        else:
            online_im = None

        if online_im is not None:
            # cv2.imshow('online_im', online_im)
            plt.cla()
            if tracker_args['debug']:
                plt.subplot(2,2,1)
                plt.imshow(cv2.cvtColor(online_im,cv2.COLOR_BGR2RGB))
                plt.axis('off')

                if len(history_im) >= 1:
                    plt.subplot(2, 2, 2)
                    plt.imshow(cv2.cvtColor(history_im[-1], cv2.COLOR_BGR2RGB))
                    plt.axis('off')

                if len(history_im) >= 2:
                    plt.subplot(2, 2, 3)
                    plt.imshow(cv2.cvtColor(history_im[-2], cv2.COLOR_BGR2RGB))
                    plt.axis('off')

                if len(history_im) >= 3:
                    plt.subplot(2, 2, 4)
                    plt.imshow(cv2.cvtColor(history_im[-3], cv2.COLOR_BGR2RGB))
                    plt.axis('off')
            else:
                plt.imshow(cv2.cvtColor(online_im,cv2.COLOR_BGR2RGB))
                plt.axis('off')
            #plt.savefig('./track_results/temp.png')
            plt.pause(0.01)
            history_im.append(online_im)

        if im_track_save_dir is not None:
            if online_im is None:
                raise ValueError('Tracking image want to be saved, but the image is None!')
            cv2.imwrite(os.path.join(im_track_save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

    # save results
    if not os.path.exists(os.path.dirname(result_filename)):
        os.makedirs(os.path.dirname(result_filename))
    write_results(result_filename, results, data_type)

    # change images to a video
    if im_track_save_dir is not None:
        images2video(im_dir=im_track_save_dir, fps=5)

    return timer.total_time


def do_mot_eval(year, phase, seq_names=None, det_root=None, result_path=None,
                save_image=False, plot_type='tracks',
                anchor_id_to_show=1, tracker_args={}):
    if phase in ['val', 'train']:
        data_root = os.path.join(MOT_info['base_path'][year]['im'], 'train')
        label_root = os.path.join(MOT_info['base_path'][year]['label'], 'train')
    elif phase in ['test']:
        data_root = os.path.join(MOT_info['base_path'][year]['im'], 'test')
        label_root = os.path.join(MOT_info['base_path'][year]['label'], 'test')
    else:
        raise ValueError('Unknown phase {}, it should be one of train, test, val'.format(phase))

    if seq_names is None:
        seqs = MOT_info['sequences'][year][phase]
        detector_names = MOT_info['detectors'][year][phase]
        seq_names = []
        for seq in seqs:
            for det in detector_names:
                if det.strip() != '':
                    seq_full_name = seq + '-' + det.strip()
                else:
                    seq_full_name = seq
                seq_names.append(seq_full_name)

    if result_path is None:
        result_path = os.path.join('track_results')

    data_type = 'mot'

    # run tracking
    accs = []
    total_time = 0
    for seq in seq_names:
        if 'MOT17Det' in data_root:
            seq_tmp = '-'.join(seq.split('-')[0:2]) # remove the detector name in MOT17 seqs
            seq_tmp = seq_tmp.replace('MOT16', 'MOT17') # change the seq name of MOT16 sequences in MOT17Det
            im_track_save_dir = os.path.join(data_root, seq_tmp, 'img_track') if save_image else None
        else:
            im_track_save_dir = os.path.join(data_root, seq, 'img_track') if save_image else None

        print('start seq: {}'.format(seq))

        # check if there is a detector
        if tracker_args['detection_model'] is not None and ('MOT16' in seq or 'DPM' in seq):
            # for DPM detections, use the MOT16-det-raw detections
            label_root_tmp = MOT_info['base_path']['MOT16-det-dpm-raw']
        else:
            label_root_tmp = label_root

        loader = get_seq_loader(data_root=data_root, label_root=label_root_tmp, seq_name=seq,
                                phase=phase,
                                det_root=det_root, use_flow=tracker_args['use_flow'])
        result_filename = os.path.join(result_path, '{}.txt'.format(seq))

        seq_time = eval_seq(loader, data_type, result_filename,
                             im_track_save_dir=im_track_save_dir, plot_type=plot_type,
                             anchor_id_to_show=anchor_id_to_show, tracker_args=tracker_args)
        total_time += seq_time
        print('Evaluate seq: {}, total time: {}'.format(seq, total_time))
        evaluator = Evaluator(data_root=label_root, seq_name=seq, data_type=data_type, phase=phase)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    metrics = mm.metrics.motchallenge_metrics
    # metrics = None
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seq_names, metrics)
    overall_mota = float(summary['mota']['OVERALL'])
    Evaluator.save_summary(summary, os.path.join(result_path, 'summary_{}_{}.xlsx'.format(year, phase)))

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    with open(os.path.join(result_path, 'summary_{}_{}.txt'.format(year, phase)), 'w') as f:
        print(strsummary, file=f)
        f.close()

    return overall_mota, strsummary


if __name__ == '__main__':

    from tracker_config import get_config
    tracker_config = get_config()

    overall_mota = do_mot_eval(year=tracker_config['year'],
                               phase=tracker_config['phase'],
                               det_root=tracker_config['det_root'],
                               result_path=tracker_config['track_result_path'],
                               plot_type=tracker_config['plot_type'],
                               anchor_id_to_show=tracker_config['anchor_id_to_show'],
                               save_image=tracker_config['save_image'],
                               tracker_args=tracker_config['init_args'])
