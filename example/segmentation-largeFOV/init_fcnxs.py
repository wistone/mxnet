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

def init_from_vgg16(ctx, largeFOV_symbol, vgg16fc_args, vgg16fc_auxs):
    largeFOV_args = vgg16fc_args.copy()
    largeFOV_auxs = vgg16fc_auxs.copy()
    for k,v in largeFOV_args.items():
        if(v.context != ctx):
            largeFOV_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(largeFOV_args[k])
    for k,v in largeFOV_auxs.items():
        if(v.context != ctx):
            largeFOV_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(largeFOV_auxs[k])

    data_shape=(1,3,321,321)
    arg_names = largeFOV_symbol.list_arguments()
    arg_shapes, _, _ = largeFOV_symbol.infer_shape(data=data_shape)
    
    rest_bias_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_bias']])
    largeFOV_args.update(rest_bias_params)

    rest_weight_params = dict([(x[0], mx.random.normal(0, 0.01, x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_weight']])
    largeFOV_args.update(rest_weight_params)

    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["bigscore_weight"]])

    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        largeFOV_args[k] = mx.nd.array(initw, ctx)
    return largeFOV_args, largeFOV_auxs


