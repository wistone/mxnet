# pylint: skip-file
import mxnet as mx

def vgg16_pool3(input, workspace_default=1024):
    # group 1
    conv1_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=64,
                workspace=workspace_default, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64,
                workspace=workspace_default, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(3, 3), stride=(2,2), pad=(1,1), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                workspace=workspace_default, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128,
                workspace=workspace_default, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(3, 3), stride=(2,2), pad=(1,1), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256,
                workspace=workspace_default, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(data=relu3_3, pool_type="max", kernel=(3, 3), stride=(2,2), pad=(1,1), name="pool3")
    return pool3

def vgg16_pool4(input, workspace_default=1024):
    # group 4
    conv4_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
                workspace=workspace_default, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(data=relu4_3, pool_type="max", kernel=(3, 3), stride=(1,1), pad=(1, 1), name="pool4")
    return pool4

def vgg16_score(input, numclass, workspace_default=1024):
    # group 5
    conv5_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(2, 2), dilate=(2, 2), num_filter=512,
                workspace=workspace_default, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(2, 2), dilate=(2, 2), num_filter=512,
                workspace=workspace_default, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="conv1_2")
    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(2, 2), dilate=(2, 2), num_filter=512,
                workspace=workspace_default, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1,1), pad=(1,1), name="pool5")
    pool5a = mx.symbol.Pooling(data=pool5, pool_type="avg", kernel=(3, 3), stride=(1,1), pad=(1,1), name="pool5a")
    # group 6
    fc6 = mx.symbol.Convolution(data=pool5a, kernel=(3, 3), num_filter=1024, pad=(12,12), dilate=(12,12),
                workspace=workspace_default, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.Convolution(data=drop6, kernel=(1, 1), num_filter=1024,
                workspace=workspace_default, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    score = mx.symbol.Convolution(data=drop7, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score")
    return score

def largeFOV_score(input, numclass=21, workspace_default=1024):
    # score out
    bigscore = mx.symbol.Deconvolution(data=input, kernel=(16,16), stride=(8,8), num_filter=numclass,
                workspace=workspace_default, name="bigscore")
    softmax = mx.symbol.SoftmaxOutput(data=bigscore, multi_output=True, use_ignore=True, ignore_label=255, name="softmax")
    return softmax

def get_largeFOV_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    pool3 = vgg16_pool3(data, workspace_default)
    pool4 = vgg16_pool4(pool3, workspace_default)
    score = vgg16_score(pool4, numclass, workspace_default)
    softmax = largeFOV_score(score, numclass, workspace_default)
    return softmax

