# train fcn-32s model
python -u fcn_xs.py --model=fcn32s --prefix=/data2/jpshi/deep-net/vgg16_20M \
       --epoch=1 

## train fcn-16s model
#python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_VGG16 \
      #--epoch=31 --init-type=fcnxs

# train fcn-8s model
#python -u fcn_xs.py --model=fcn8s --prefix=FCN16s_VGG16 \
      #--epoch=27 --init-type=fcnxs
