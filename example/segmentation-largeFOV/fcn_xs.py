# pylint: skip-file
import find_mxnet
import mxnet as mx
import sys, os
import argparse
import numpy as np
import logging
import symbol_largeFOV
import init_fcnxs
from data import FileIter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)

def main():
    largeFOV = symbol_largeFOV.get_largeFOV_symbol(numclass=21, workspace_default=1536)
    fcnxs_model_prefix = "model/largeFOV"

    arg_names = largeFOV.list_arguments()
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, 1)
    fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, largeFOV, fcnxs_args, fcnxs_auxs)

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

    model = mx.model.FeedForward(
        ctx                 = mx.gpu(0),
        symbol              = largeFOV,
        num_epoch           = 6,
        learning_rate       = 0.001,
        wd                  = 0.0005,
        momentum            = 0.9,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs)

    model.fit(
        X                   = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--prefix', default='/data2/jpshi/deep-net/vgg16_20M',
        help='The prefix(include path) of vgg16 model with mxnet format.')
    args = parser.parse_args()
    logging.info(args)
    main()
