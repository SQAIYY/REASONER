import argparse
import collections
import numpy as np

from data_loader.data_loaders import *
from model_WESAD.loss import CrossEntropy_Loss, MSELoss, Feature_Level_Consistency_Loss, MultVariateKLD
import model.metric as module_metric
# import model.model_CNN_VIT1 as module_arch  # model_EEG_CNNvit_Orthogv2
import model_WESAD.model as module_arch  # model_EEG_CNNvit_Orthogv2
from parse_config import ConfigParser
from trainer.trainer_miss_WESAD import Trainer
from utils.util import *
import random
import torch
import torch.nn as nn
import os
# fix random seeds for reproducibility
SEED = 86334
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('train')

    # build model architecture, initialize weights, then print to console
    model = config.init_obj('arch', module_arch)
    # model.apply(weights_init_normal)
    logger.info(model)

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    data_loader, valid_data_loader, data_count, train_data_count = data_generator_np_miss_WESAD(folds_data[fold_id][0],
                                                                                              folds_data[fold_id][1],
                                                                                              batch_size)

    weights_for_each_class = calc_class_weightWESAD(data_count)

    criterion = CrossEntropy_Loss(ce_weight=weights_for_each_class)
    criterion_MSE_m1 = MSELoss()
    criterion_MSE_m2 = MSELoss()
    criterion_Feature_Level = Feature_Level_Consistency_Loss()
    criterion_Context_Level = MultVariateKLD(reduction='mean')
    criterion_MSE_P1 = MSELoss()
    criterion_MSE_P2 = MSELoss()

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, criterion, criterion_MSE_m1, criterion_MSE_m2, criterion_Feature_Level,
                      criterion_Context_Level, criterion_MSE_P1, criterion_MSE_P2, metrics,
                      optimizer,
                      config=config,
                      data_loader=data_loader,
                      fold_id=fold_id,
                      valid_data_loader=valid_data_loader,
                      class_weights=weights_for_each_class)

    trainer.train()


if __name__ == '__main__':


    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="0",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="1",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="2",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="3",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="4",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="5",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="6",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="7",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="8",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="9",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="10",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="11",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="12",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)


    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="13",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, default="14",
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, default="data/output_data/WESAD_00/",
                      help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data_WESDA(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)









