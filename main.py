import logging
from pathlib import Path
from functools import partial

import torch
import yaml

import os
import process
import quan
import util
from model import create_model

dataset_mapping = {'imagenet': 'ImageNet', 'cifar10': 'CIFAR10'}
architecture_mapping = {'resnet18': 'ResNet18', 'resnet20': 'ResNet20', 'resnet20_relu': 'ResNet20'}
algorithm_mapping = {'lsq': 'LSQ'}

def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')

    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    experiment_dir = '/'.join([dataset_mapping[args.dataloader.dataset],
                               architecture_mapping[args.arch],
                               algorithm_mapping[args.quan.act.mode]])
    log_dir = util.init_logger(args.name, experiment_dir, output_dir,
                               script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not torch.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = torch.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        torch.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.deterministic = True

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    # Create the model
    model = create_model(args)
    model.to(args.device.type)

    # Prepare for the quantized model
    quant_hook_handles = quan.register_hooks(model, args.quan)
    process.iteration(train_loader, model, args)
    quan.unregister_hooks(quant_hook_handles)
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    # tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')

    if args.device.gpu and not args.dataloader.serialized:
        model = torch.nn.DataParallel(model, device_ids=args.device.gpu)

    # start_epoch = 0
    # if args.resume.path:
    #     model, start_epoch, _ = util.load_checkpoint(
    #         model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device.type)
    aux_criterions = list()
    if args.quan.pact.use_pact:
        parameters = list()
        for name, parameter in model.named_parameters():
            if 'pact_alpha' in name:
                parameters.append(parameter)
        alpha_criterion = util.L2RegularizationLoss(parameters,
                                                    args.quan.pact.lambda_alpha,
                                                    torch.device(args.device.type))
        aux_criterions.append(alpha_criterion)
    if args.weight_cluster_loss.use_wc_loss:
        if args.weight_cluster_loss.mode == 'naive':
            WeightClusterLoss = util.WeightClusterLoss
        elif args.weight_cluster_loss.mode == 'topk':
            WeightClusterLoss = partial(util.TopkWeightClusterLoss,
                                        topk=args.weight_cluster_loss.topk)
        elif args.weight_cluster_loss.mode == 'soft':
            WeightClusterLoss = util.SoftWeightClusterLoss
        elif args.weight_cluster_loss.mode == 'balanced':
            WeightClusterLoss = partial(util.BalancedWeightClusterLoss,
                                        percentile=args.weight_cluster_loss.percentile)
        else:
            raise ValueError('Error!')
        qmodules = util.extract_qmodule(model)
        wc_criterion = WeightClusterLoss(modules=qmodules,
                                         coefficient=args.weight_cluster_loss.coefficient,
                                         bit=args.quan.weight.bit,
                                         per_channel=args.quan.weight.per_channel,
                                         symmetric=args.quan.weight.symmetric,
                                         device=torch.device(args.device.type))
        aux_criterions.append(wc_criterion)
    if aux_criterions:
        criterion = util.MultipleLoss(cross_entropy_loss=criterion,
                                      aux_criterions=aux_criterions)

    model_parameters = util.extract_param(model)
    model_q_parameters = util.extract_qparam(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.optimizer.learning_rate,
    #                             momentum=args.optimizer.momentum,
    #                             weight_decay=args.optimizer.weight_decay)
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
        quan_optimizer = torch.optim.SGD(model_q_parameters,
                                         lr=optimizer_params.learning_rate,
                                         momentum=optimizer_params.momentum,
                                         weight_decay=optimizer_params.weight_decay)
    elif args.quan_optimizer.mode == 'adam':
        optimizer_params = args.quan_optimizer.adam_params
        quan_optimizer = torch.optim.Adam(model_q_parameters,
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
        model, start_epoch, optimizer, quan_optimizer, _ = util.load_checkpoint(
            model, optimizer, quan_optimizer, args.resume.path, args.device.type,
            lean=args.resume.lean)

    # logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    # logger.info('LR scheduler: %s\n' % lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info(('Optimizer: %s' % quan_optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % scheduler)

    perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            # logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            logger.info(f'>>>>>>>> Epoch {start_epoch-1} (pre-trained model evaluation)')
            top1, top5, _ = process.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            # t_top1, t_top5, t_loss = process.train(train_loader, model, criterion,
            #                                        optimizer, lr_scheduler, epoch,
            #                                        monitors, args)
            t_top1, t_top5, t_loss = process.train(train_loader, model, criterion,
                                                   optimizer, quan_optimizer, scheduler,
                                                   quan_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss = process.validate(val_loader, model, criterion, epoch,
                                                      monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, optimizer, quan_optimizer,
                                 {'top1': v_top1, 'top5': v_top5}, is_best, args.name,
                                 log_dir)

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args)

    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')


if __name__ == "__main__":
    main()
