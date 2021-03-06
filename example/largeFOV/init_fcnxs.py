# pylint: skip-file
import mxnet as mx
import numpy as np
import sys
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def init_from_vgg16(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
    fcnxs_args = vgg16fc_args.copy()
    fcnxs_auxs = vgg16fc_auxs.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape=(1,3,500,500)
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_weight', 'score_bias', 'score_pool4_weight', 'score_pool4_bias', \
                        'score_pool3_weight', 'score_pool3_bias']])
    fcnxs_args.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs

