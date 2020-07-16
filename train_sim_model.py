import os
import argparse

def args_parse():
    # projects description
    desc = "Graph Similarity"
    parser = argparse.ArgumentParser(description=desc)

    # cuda config
    parser.add_argument('--gpu_id', default='0', type=str,
                        help="the ID of GPUS to use, such as '0', or '0,1'")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help="Use cuda?")

    # training config
    parser.add_argument('--save_dir', type=str, default='logs',  # required=True,
                        help='Directory name to save the model')
    parser.add_argument('--config_file', type=str, default='',
                        help='the path of configure file')
    parser.add_argument('--session', type=int, default=0, help='The session of this experiment')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='The start epoch')
    parser.add_argument('--valid_epochs', type=int, default=1, help='How many epochs to validate the model once')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of works to load data')

    # resume a training process, the training process will be eactly restarted
    parser.add_argument('--resume', action='store_true', default=False, # False,
                        help="continue to train? If true, the saved optimizer, model parameter, "
                             "and scheduler will be used to replay the training process")
    parser.add_argument('--resume_epoch', type=int, default=20, # -1,
                        help='which epoch is used to resume?')

    # start a new training process but with a pretrainind model parameter weight
    parser.add_argument('--para_path', type=str, default='', # '/home/liuqk/Program/pycharm/MOT-graph/logs/GraphSimilarity_v5/1/GraphSimilarity_v5_1_20.pth',
                        help='start a new training process but with a pretrainind model parameter weight')

    parser.add_argument('--model_name', type=str, default='GraphSimilarity_v5',  # 'model',
                        help='the name of this session')

    # optimizers
    parser.add_argument('--optim', type=str, default='adam', choices=["rmsprop", "adam"])
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0001)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--scheduler', type=str, default='exp', choices=["exp", "cosine"],
                        help="Learning rate scheduler type")

    # tensorboardX
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='use tensorboardX to visualize the training process')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.start_epoch = 0
    args.save_dir = os.path.join(os.getcwd(), args.save_dir, args.model_name, str(args.session))

    return args

import pdb
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tensorboardX import SummaryWriter
from lib.models.config.config import get_config, merge_arg_to_config, save_config, load_config
from lib.utils.io import my_print
from lib.datasets.dataset_utils import load_mot_tracklet_loader
from lib.datasets.mot_tracklet import MOTFramePair
from lib.models.graph.similarity_model import GraphSimilarityl
from lib.models.net_utils import load_model_para


# if not torch.set_flush_denormal(True):
#     print("Unable to set flush denormal")
#     print("Pytorch compiled without advanced CPU")
#     print("at: https://github.com/pytorch/pytorch/blob/84b275b70f73d5fd311f62614bccc405f3d5bfa3/aten/src/ATen/cpu/FlushDenormal.cpp#L13")


def get_optimizer(model, optimizer_name, lr, decay,
                  momentum, eps):
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            print('train parameter: {}'.format(key))
            par = {'params': [value], 'lr': lr, 'weight_decay': decay}
            if optimizer_name == 'rmsprop':
                par['momentum'] = momentum
                par['eps'] = eps
            params += [par]

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(params)

    return optimizer


def get_scheduler(optim, schedule_type, step_size, t_max):
    """get the scheduler

    Args:
        optim: the optimizer
        schedule_type: str, the type of scheduler
        step_size: used for StepLR, the schduler makes a step per iter
        t_max: used for CosineAnnealingLR, the total number of iter

    """
    if schedule_type == "exp":
        return StepLR(optim, step_size, 0.5)
    elif schedule_type == "cosine":
        return CosineAnnealingLR(optim, t_max)
    else:
        return None


def get_model_para_path(save_dir, model_name, session, epoch, epochs):
    para_path = os.path.join(save_dir, '{}_{}_{}.pth'.format(model_name, session, str(epoch).zfill(len(str(epochs)))))
    return para_path


def parse_tensor_dict(input_dict, name='loss'):
    """This function parse the dict and return the total value as well as a string description.
    Each element in the dict is a tensor
    """
    descript = ''
    total = 0
    for k in input_dict.keys():
        if name in k:
            total = total + input_dict[k]
            descript = descript + ', ' + '{}: {}'.format(k, round(input_dict[k].mean().item(), 3))
    descript = descript[2:] # remove the first ', '
    return total.mean(), descript


if __name__ == '__main__':

    # handle args and config
    args = args_parse()

    config_file = os.path.join(args.save_dir, 'config.json')
    if not args.resume:
        if args.config_file == '':
            config = get_config()
        else:
            config = load_config(cfg_path=args.config_file)
    else:
        config = load_config(cfg_path=config_file)
    config = merge_arg_to_config(args=args, cfg=config)

    # check the version of model name
    if 'Graph' in args.model_name:
        version = args.model_name.split('_')[-1]
        assert version in config['GraphSimilarity']['init_args']['match_name'] and 'GraphMatch' in config['GraphSimilarity']['init_args']['match_name']
    elif 'Naive' in args.model_name:
        assert 'NaiveMatch' in config['GraphSimilarity']['init_args']['match_name']

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_file = os.path.join(args.save_dir, 'logs.txt')
    if args.resume:
        log_file = open(log_file, 'a')
    else:
        log_file = open(log_file, 'w')
        my_print(args, file=log_file)
        my_print(config, file=log_file)

    # save config
    save_config(cfg=config, cfg_path=config_file)

    # tensorboard to show
    if not args.no_tensorboard:
        borad_log_dir = os.path.join(args.save_dir, 'borad_log')
        if not os.path.exists(borad_log_dir):
            os.makedirs(borad_log_dir)
        board_writer = SummaryWriter(log_dir=borad_log_dir)

    # get the dataset loaders
    train_loader, val_loader = load_mot_tracklet_loader(dataset=MOTFramePair,
                                                        dataset_config=config['MOTFramePair'],
                                                        batch_size=args.batch_size,
                                                        shuffle=False, # set false here
                                                        num_workers=args.num_workers)

    # get the models, optimizer, scheduler and the loss model
    #pdb.set_trace()
    model = GraphSimilarityl(**config['GraphSimilarity']['init_args'])

    optim = get_optimizer(model=model, optimizer_name=args.optim,
                          lr=args.lr, decay=args.decay, momentum=args.momentum, eps=args.eps)

    scheduler = get_scheduler(optim=optim, schedule_type=args.scheduler,
                              step_size=int(int(args.epochs/3) * len(train_loader)), t_max=args.epochs * len(train_loader))

    if torch.cuda.is_available() and not args.no_cuda:
        torch_device = torch.device('cuda')
        model = nn.DataParallel(model).to(torch_device)
    else:
        torch_device = torch.device('cpu')
        model = model.to(torch_device)

    # start a new training process but with a pretrainind model parameter weight
    if args.para_path.strip() != '':
        state = torch.load(args.para_path, map_location=torch.device('cpu'))
        # model.load_state_dict(state['model'])
        load_model_para(state['model'], model)
        my_print('load model weight from {}'.format(args.para_path), file=log_file)

    # resume a training procedure
    if args.resume:
        para_path = get_model_para_path(save_dir=args.save_dir, model_name=args.model_name,
                                        session=args.session, epoch=args.resume_epoch,
                                        epochs=args.epochs)
        state = torch.load(para_path, map_location=torch_device)
        load_model_para(state['model'], model)
        # model.load_state_dict(state['model'])
        optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        args.start_epoch = state['epoch'] + 1
        my_print('resume the training for {}'.format(para_path), file=log_file)

    # begin to train the models
    val_loader.dataset.prepare_data(batch_size=args.batch_size, shuffle=False) # validation loader only needs to prepared once

    #with torch.autograd.set_detect_anomaly(True):

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        # reload the data
        train_loader.dataset.prepare_data(batch_size=args.batch_size, epoch=epoch, shuffle=True) # shuffle here
        t_start = time.time()
        loss_his = []
        acc_his = []
        for itr, batch_data in enumerate(train_loader):
            # scheduler.step()

            batch_data = list(batch_data)
            batch_data = [data.to(torch_device) for data in batch_data]
            track_ims, det_ims, track_ids, det_ids, track_boxes, det_boxes, im_shape = tuple(batch_data)

            loss, acc = model(track_ims, det_ims, track_ids, det_ids, track_boxes, det_boxes, im_shape)
            if len(loss.keys()) == 0 or len(acc.keys()) == 0:
                continue

            loss_total, loss_str = parse_tensor_dict(loss, 'loss')
            acc_total, acc_str = parse_tensor_dict(acc, 'acc')

            optim.zero_grad()
            loss_total.mean().backward()
            optim.step()

            scheduler.step()

            loss_his.append(loss_total.mean().item())
            acc_his.append(acc_total.mean().item())

            if itr % 10 == 0:
                t = time.time() - t_start
                cur_lr = optim.param_groups[-1]['lr']
                my_print('[Train {}] Epoch {}/{}: {}/{}, loss: {}, acc: {}, lr: {}, time: {}'.format(args.session, epoch,
                         args.epochs, itr, len(train_loader), round(sum(loss_his)/len(loss_his), 3),
                         round(sum(acc_his)/len(acc_his), 3), cur_lr, round(t, 3)),
                        file = log_file)
                my_print('\t\t{}, {}'.format(loss_str, acc_str), file=log_file)
                loss_his = []
                acc_his = []
                t_start = time.time()

            if not args.no_tensorboard:
                board_writer.add_scalar('data/learning_rate', cur_lr, itr)
                board_writer.add_scalar('loss/loss', loss_total.mean().item(), itr)
                board_writer.add_scalar('accuracy/acc', acc_total.mean().item(), itr)
                for k in loss.keys():
                    board_writer.add_scalar('loss/{}'.format(k), loss[k].mean().item(), itr)
                for k in acc.keys():
                    board_writer.add_scalar('accuracy/{}'.format(k), acc[k].mean().item(), itr)

        # save the model and add weights to tensorboard
        if not args.no_tensorboard:
            for name, param in model.named_parameters():
                board_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        para_path = get_model_para_path(save_dir=args.save_dir, model_name=args.model_name,
                                        session=args.session, epoch=epoch, epochs=args.epochs)
        state = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, para_path)
        my_print('Model saved in {}'.format(para_path), log_file)

        # validate
        if epoch % args.valid_epochs == 0:
            loss_his = []
            acc_his = []
            t_start = time.time()
            model.eval()
            for itr, batch_data in enumerate(val_loader):
                batch_data = list(batch_data)
                batch_data = [data.to(torch_device) for data in batch_data]
                track_ims, det_ims, track_ids, det_ids, track_boxes, det_boxes, im_shape = tuple(batch_data)
                with torch.no_grad():
                    loss, acc = model(track_ims, det_ims, track_ids, det_ids, track_boxes, det_boxes, im_shape)

                loss_total, loss_str = parse_tensor_dict(loss, 'loss')
                if len(loss.keys()) == 0 or len(acc.keys()) == 0:
                    continue

                acc_total, acc_str = parse_tensor_dict(acc, 'acc')

                loss_his.append(loss_total.mean().item())
                acc_his.append(acc_total.mean().item())

                if itr % 10 == 0:
                    t = time.time() - t_start
                    my_print('[valid {}] Epoch {}/{}: {}/{}, loss: {}, acc: {}, lr: {}, time: {}'.format(args.session,
                              epoch, args.epochs, itr, len(val_loader), round(sum(loss_his) / len(loss_his), 3),
                              round(sum(acc_his) / len(acc_his), 3), cur_lr, round(t, 3)), file=log_file)
                    my_print('\t\t{}, {}'.format(loss_str, acc_str), file=log_file)
                    loss_his = []
                    acc_his = []
                    t_start = time.time()

    log_file.close()
    if not args.no_tensorboard:
        board_writer.close()

























