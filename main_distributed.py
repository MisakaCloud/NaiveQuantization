# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import logging
import yaml
import numpy as np
import os.path as osp
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision as tv

import process
import quan
import util
from model import create_model

dataset_mapping = {'imagenet': 'ImageNet', 'cifar10': 'CIFAR10'}
architecture_mapping = {'resnet18': 'ResNet18', 'resnet20': 'ResNet20', 'resnet20_relu': 'ResNet20'}
algorithm_mapping = {'lsq': 'LSQ', 'lsqbeta': 'LSQ', 'lcq': 'LCQ'}


def main_worker(gpu, ngpus_per_node, args):
    # dump experiment config
    with open(osp.join(args.log_dir,'args.yaml'), 'w') as yaml_file:
        yaml.safe_dump(args, yaml_file)

    if args.multiprocessing_distributed:
        args.rank = 0 * ngpus_per_node + gpu
        dist.init_process_group(backend=args.distributed.dist_backend,
                                init_method=args.distributed.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    logger = util.setup_logger(args.log_dir, args.rank)
    logger.info(f'Use GPU: {gpu} for training')
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        logger.warning('You have chosen to seed training. '
                       'This will turn on the CUDNN deterministic setting, '
                       'which can slow down your training considerably! '
                       'You may see unexpected behavior when restarting '
                       'from checkpoints.')

    # if args.rank == 0:
    #     tbmonitor = util.TensorBoardMonitor(logger, args.log_dir)
    #     logger.info('TensorBoard data directory: %s/tb_runs' % args.log_dir)
    # else:
    #     tbmonitor = None

    # Initialize data loader
    dataloaders = util.load_data(args.dataloader, args.multiprocessing_distributed)
    train_loader, val_loader, test_loader = dataloaders
    logger.info(f'Dataset `{args.dataloader.dataset}` size:' +
                f'\n\t\t  Training Set = {len(train_loader.sampler)} ({len(train_loader)})' +
                f'\n\t\tValidation Set = {len(val_loader.sampler)} ({len(val_loader)})' +
                f'\n\t\t      Test Set = {len(test_loader.sampler)} ({len(test_loader)})')

    # Create the model
    model = create_model(args)
    if args.multiprocessing_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Prepare for the quantized model
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    logger.info('Inserted quantizers into the original model')

    if args.multiprocessing_distributed:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
        batch_size = args.dataloader.batch_size
        workers = args.dataloader.workers
        args.dataloader.batch_size = int(batch_size / ngpus_per_node)
        args.dataloader.workers = int((workers+ngpus_per_node-1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device.type)

    model_parameters = util.extract_param(model)
    model_qparameters = util.extract_qparam(model)
    model_quantizers = util.extract_quantizer(model)
    if args.optimizer.mode == 'sgd':
        optimizer_params = args.optimizer.sgd_params
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optimizer_params.learning_rate,
                                    momentum=optimizer_params.momentum,
                                    weight_decay=optimizer_params.weight_decay)
    elif args.optimizer.mode == 'adam':
        optimizer_params = args.optimizer.adam_params
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optimizer_params.learning_rate,
                                     betas=optimizer_params.betas,
                                     weight_decay=optimizer_params.weight_decay,
                                     amsgrad=optimizer_params.amsgrad)
    else:
        raise ValueError(f'Invalid optimizer type {args.optimizer.mode} in configs.')
    if args.quan_optimizer.mode == 'sgd':
        optimizer_params = args.quan_optimizer.sgd_params
        quan_optimizer = torch.optim.SGD(model_qparameters,
                                         lr=optimizer_params.learning_rate,
                                         momentum=optimizer_params.momentum,
                                         weight_decay=optimizer_params.weight_decay)
    elif args.quan_optimizer.mode == 'adam':
        optimizer_params = args.quan_optimizer.adam_params
        quan_optimizer = torch.optim.Adam(model_qparameters,
                                          lr=optimizer_params.learning_rate,
                                          betas=optimizer_params.betas,
                                          weight_decay=optimizer_params.weight_decay,
                                          amsgrad=optimizer_params.amsgrad)
    else:
        raise ValueError(f'Invalid optimizer type {args.quan_optimizer.mode} in configs.')
    scheduler = util.lr_scheduler(optimizer,
                                  batch_size=train_loader.batch_size,
                                  num_samples=len(train_loader.sampler),
                                  **args.lr_scheduler)
    quan_scheduler = util.lr_scheduler(quan_optimizer,
                                       batch_size=train_loader.batch_size,
                                       num_samples=len(train_loader.sampler),
                                       **args.lr_scheduler)

    start_epoch = 0
    if args.resume.path:
        logger.info(f'>>>>>>>> Loading checkpoint \'{args.resume.path}\'')
        model_device = f'cuda:{gpu}'
        model, start_epoch, optimizer, quan_optimizer, _ = util.load_checkpoint(
            model, optimizer, quan_optimizer, args.resume.path, model_device,
            lean=args.resume.lean)
        logger.info(f'>>>>>>>> Loaded checkpoint \'{args.resume.path}\'')

    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info(('Optimizer: %s' % quan_optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % scheduler)

    if args.rank == 0:
        perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info(f'>>>>>>>> Epoch {start_epoch-1}'
                         '(pre-trained model evaluation)')
            # top1, top5, _ = process.validate(val_loader, model, criterion,
            #                                  start_epoch - 1, args)
            # if args.rank == 0:
            #     perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            if args.multiprocessing_distributed:
                train_loader.sampler.set_epoch(epoch)

            t_top1, t_top5, t_loss = process.train(train_loader, model, model_quantizers,
                                                   criterion, optimizer, quan_optimizer,
                                                   scheduler, quan_scheduler, epoch, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch,
                                                      args)

            if args.rank == 0:
                # tbmonitor.writer.add_scalars('Train_vs_Val/Loss_rank',
                #                              {'train': t_loss, 'val': v_loss}, epoch)
                # tbmonitor.writer.add_scalars('Train_vs_Val/Top1_rank',
                #                              {'train': t_top1, 'val': v_top1}, epoch)
                # tbmonitor.writer.add_scalars('Train_vs_Val/Top5_rank',
                #                              {'train': t_top5, 'val': v_top5}, epoch)

                perf_scoreboard.update(v_top1, v_top5, epoch)
                is_best = perf_scoreboard.is_best(epoch)
                util.save_checkpoint(epoch, args.arch, model, optimizer, quan_optimizer,
                                     {'top1': v_top1, 'top5': v_top5}, is_best, args.name,
                                     args.log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, args)

    # if args.rank == 0:
    #     tbmonitor.writer.close()  # close the TensorBoard


def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    experiment_dir = '/'.join([dataset_mapping[args.dataloader.dataset],
                               architecture_mapping[args.arch],
                               algorithm_mapping[args.quan.act.mode]])
    log_dir = util.init_directory(args.name, experiment_dir, output_dir)
    args.log_dir = str(log_dir)

    if args.device.type == 'cpu' or not torch.cuda.is_available():
        raise NotImplementedError('Code can only run on GPU device. '
                                  'However, no available GPU device detected')

    node_num = 1
    ngpus_per_node = torch.cuda.device_count()
    args.multiprocessing_distributed = ngpus_per_node > 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * node_num
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        main_worker(0, ngpus_per_node, args)

    # logger = util.setup_logger(args.log_dir, 0)
    # logger.info('Program completed successfully ... exiting ...')
    # logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    print('Program completed successfully ... exiting ...')
    print('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')


if __name__ == '__main__':
    main()

