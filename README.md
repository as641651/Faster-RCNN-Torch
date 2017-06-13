# Faster-RCNN-Torch
Torch version of Faster RCNN model with ROI and Bilinear ROI Pooling of region proposals. Essential modules have been adapted from the [Densecap](https://github.com/jcjohnson/densecap) repository. This work was carried out with the department [Informatik6](https://www-i6.informatik.rwth-aachen.de/) at [RWTH Aachen](http://www.rwth-aachen.de/cms/~a/root/?lidx=1) university under the supervision of [Mr.Harald Hanselmann, M.Sc](http://www.informatik.rwth-aachen.de/cms/Informatik/Fachgruppe/Kontakt/Fachstudienberater/~muxq/Harald-Hanselmann-M-Sc-Fachstudienber/?lidx=1&allou=1)

## Dependencies

### Required:
1) [Torch](http://torch.ch/)
```bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
```

2) After installing torch, you can install / update these dependencies by running the following:
```bash
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
luarocks install lua-cjson
luarocks install hdf5
luarocks install cv #Requires OpenCV 3.1 TODO: Remove this and use the torch image module
```
### Optional:

To use bilinear ROI pooling:
```bash
luarocks install stnbhwd
```

## Pre-trained models for initialization

Only VGG16 and VGG1024 models are currently supported. To convert pre-trained caffe versions of imagenet or py-faster-rcnn models to lua tables, see here. Alternatively, you can download the following torch compatable versions :

[Imagenet (VGG16 + FCN)](https://drive.google.com/open?id=0B8Uc-OssxXlDbmJKWjk4MU9QaWs) <br />
[FasterRCNN (VGG16 + RPN + FCN)](https://drive.google.com/open?id=0B8Uc-OssxXlDRXQ4WC1xQ1JoT28)

After creating / downloading the torch version of pretrained model, set the corresponding model path in 'config.lua' :
```bash
init_model.vgg16 = "" #Imagenet VGG16
init_model.frcnn_vgg16 = "init_models/frcnn_vgg16.t7" #Faster RCNN VGG16
init_model.vgg1024 = "" #Imagenet VGG1024
init_model.frcnn_vgg1024 = "" #Faster RCNN VGG1024
```
## Running the script

The script 'run.lua' is the starting point for our object detection task. To see the available options, hit
```bash
th run.lua -h
```

### Examples
1)To train a faster-rcnn VGG16 model,

* initialized with imagenet VGG16 model.  
* with usual ROI pooling of region proposals used in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). 
* for 100K iterations stepping down the learning by 0.1 every 50K iterations.
* writing checkpoint every 10K iterations to 'checkpoint.t7'.

```bash
th run.lua -max_iters 100000 -step 50000 -gamma 0.1 -save_checkpoint_every 10000 -checkpoint_path checkpoint.t7 -seed 1432
```
2) To fine-tune the model by initiliazing with the caffe trained Faster-RCNN VGG16 model, use the option **-init_rpn**. Before using this option, make sure that the torch Faster RCNN VGG16 model has been created and the path has been set in config.lua 
```bash
th run.lua -init_rpn 
```
3) To use bilinear ROI pooling on imagenet initialized VGG16 model, 
```bash
th run.lua -bilinear
```
4) To continue from a checkpoint,
```bash
th run.lua -checkpoint_start_from checkpoint.t7 -bilinear
```
5) To use caffe trained faster rcnn **VGG1024** model
```bash
th run.lua -init_rpn -vgg1024
```
6) To run only the evaluation of a saved checkpoint
```bash
th run.lua -checkpoint_start_from checkpoint.t7 -eval
```

### Performance
The mAP on Pascal VOC 2007 test set for the pre-trained Faster-RCNN VGG16 model (i.e, trained for 100K iteration with py-faster-rcnn achieveing 70%) in torch is 69.1%. 

To run only the evaluation of caffe trained faster-rcnn VGG16 model (make sure that the torch compatable Faster-RCNN VGG16 model is available) 
```bash
th run.lua -init_rpn -eval
```
Training or further finetuning caffe models in torch currently does not improve this performance.



