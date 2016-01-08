# pylint: skip-file
import sys, os
import argparse
import find_mxnet
import mxnet as mx
import numpy as np
import logging
import symbol_fcnxs
import init_fcnxs
from data import FileIter
from solver import Solver

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(11)

def main():
    fcnxs = symbol_fcnxs.get_fcn32s_symbol(numclass=21, workspace_default=1536)
    print fcnxs
    fcnxs_model_prefix = "model_pascal/FCN32s_VGG16"
    arg_names = fcnxs.list_arguments()
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, args.epoch)
    fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    train_dataiter = FileIter(
        root_dir             = "/data2/jpshi/VOC2012/",
        flist_name           = "/data2/jpshi/VOCdevkit/VOC2012/list/train_aug.txt",
        cut_off_size         = 321,
        batch_size           = 2,
        mirror               = 1,
        shuffle              = 0,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = "/data2/jpshi/VOC2012/",
        flist_name           = "/data2/jpshi/VOCdevkit/VOC2012/list/val.txt",
        cut_off_size         = 513,
        batch_size           = 1,
        mirror               = 0,
        shuffle              = 0,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = 0,
        num_epoch           = 50,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs,
        learning_rate       = 0.001,
        momentum            = 0.9,
        wd                  = 0.0005)
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcnxs',
        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
        help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=74,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    main()
