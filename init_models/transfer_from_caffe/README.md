# Caffe to Torch
The scripts ***load_vgg16.py*** and ***load_vgg1024.py*** can be used to change a model trained in caffe to lua table.

## Usage

To transfer weights from CNN + FCN layers of VGG16 model and write to a lua table in vgg16.t7,
```bash
python load_vgg16.py -m vgg16.caffemodel -o vgg16.t7
```
To transfer weights from CNN+RPN+FCN,
```bash
python load_vgg16.py -m vgg16.caffemodel -o vgg16.t7 --rpn
```
After creating the torch version, the path of this pre-trained model have to be updated in 'config.lua'



