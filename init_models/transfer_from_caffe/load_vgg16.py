import caffe
import PyTorch
from PyTorchAug import nn
import PyTorchAug
import argparse, os

def transferW(b,a):
  print b.size()
  #print a.shape
  for i in range(b.size()[0]):
     for j in range(b.size()[1]):
        b[i][j] = a[i][j]

def transferB(b,a):
  print b.size()
  #print a.shape
  for i in range(b.size()[0]):
      b[i] = a[i]

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-m',
              dest='model',
	      default='',
	      help='Path to caffe model file')
	parser.add_argument('-o',
              dest='out',
	      default='',
	      help='Path to write torch model file')
	parser.add_argument('--rpn',
              dest='rpn',
	      default=False,
              action='store_true',
	      help='Transfer RPN weights (The caffe model should have RPN)')

	  # OPTIONS
	args = parser.parse_args()
        if not args.model:
          print "No model file given"
          exit(-1)

        if not args.out:
          print "No out file path given"
          exit(-1)

        proto = 'vgg16.prototxt'
        if args.rpn:
          print "Transfering RPN layers"
          proto = 'frcnn_vgg16.prototxt'

	tnet = {}

	#net = caffe.Net('test_c.prototxt','/work/cv3/sankaran/caffe-exp/faster-rcnn-exp/VGG16_e2e_pascal/models/ms_tv_vgg16_faster_rcnn_iter_100000.caffemodel',caffe.TEST)
	net = caffe.Net(proto,args.model,caffe.TEST)
	print net.params['fc6'][0].data.shape
	print net.params['fc6'][1].data.shape

	tnet["conv1_1"] = nn.SpatialConvolutionMM(3,64,3,3,1,1,1,1)
        print "conv1_1"
	transferW(tnet["conv1_1"].weight, net.params['conv1_1'][0].data.reshape(64,27))
	transferB(tnet["conv1_1"].bias, net.params['conv1_1'][1].data)
	tnet["relu1_1"] = nn.ReLU()

	tnet["conv1_2"] = nn.SpatialConvolutionMM(64,64,3,3,1,1,1,1)
        print "conv1_2"
	transferW(tnet["conv1_2"].weight, net.params['conv1_2'][0].data.reshape(64,576))
	transferB(tnet["conv1_2"].bias, net.params['conv1_2'][1].data)
	tnet["relu1_2"] = nn.ReLU()
	tnet["pool1"] = nn.SpatialMaxPooling(2,2,2,2,0,0)

	tnet["conv2_1"]= nn.SpatialConvolutionMM(64,128,3,3,1,1,1,1)
        print "conv2_1"
	transferW(tnet["conv2_1"].weight, net.params['conv2_1'][0].data.reshape(128,576))
	transferB(tnet["conv2_1"].bias, net.params['conv2_1'][1].data)
	tnet["relu2_1"] = nn.ReLU()

	tnet["conv2_2"]= nn.SpatialConvolutionMM(128,128,3,3,1,1,1,1)
        print "conv2_2"
	transferW(tnet["conv2_2"].weight, net.params['conv2_2'][0].data.reshape(128,1152))
	transferB(tnet["conv2_2"].bias, net.params['conv2_2'][1].data)
	tnet["relu2_2"] = nn.ReLU()
	tnet["pool2"] = nn.SpatialMaxPooling(2,2,2,2,0,0)

	tnet["conv3_1"]= nn.SpatialConvolutionMM(128,256,3,3,1,1,1,1)
        print "conv3_1"
	transferW(tnet["conv3_1"].weight, net.params['conv3_1'][0].data.reshape(256,1152))
	transferB(tnet["conv3_1"].bias, net.params['conv3_1'][1].data)
	tnet["relu3_1"] = nn.ReLU()

	tnet["conv3_2"]= nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)
        print "conv3_2"
	transferW(tnet["conv3_2"].weight, net.params['conv3_2'][0].data.reshape(256,2304))
	transferB(tnet["conv3_2"].bias, net.params['conv3_2'][1].data)
	tnet["relu3_2"] = nn.ReLU()

	tnet["conv3_3"]= nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1)
        print "conv3_3"
	transferW(tnet["conv3_3"].weight, net.params['conv3_3'][0].data.reshape(256,2304))
	transferB(tnet["conv3_3"].bias, net.params['conv3_3'][1].data)
	tnet["relu3_3"] = nn.ReLU()
	tnet["pool3"] = nn.SpatialMaxPooling(2,2,2,2,0,0)

	tnet["conv4_1"]= nn.SpatialConvolutionMM(256,512,3,3,1,1,1,1)
        print "conv4_1"
	transferW(tnet["conv4_1"].weight, net.params['conv4_1'][0].data.reshape(512,2304))
	transferB(tnet["conv4_1"].bias, net.params['conv4_1'][1].data)
	tnet["relu4_1"] = nn.ReLU()

	tnet["conv4_2"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv4_2"
	transferW(tnet["conv4_2"].weight, net.params['conv4_2'][0].data.reshape(512,4608))
	transferB(tnet["conv4_2"].bias, net.params['conv4_2'][1].data)
	tnet["relu4_2"] = nn.ReLU()

	tnet["conv4_3"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv4_3"
	transferW(tnet["conv4_3"].weight, net.params['conv4_3'][0].data.reshape(512,4608))
	transferB(tnet["conv4_3"].bias, net.params['conv4_3'][1].data)
	tnet["relu4_3"] = nn.ReLU()
	tnet["pool4"] = nn.SpatialMaxPooling(2,2,2,2,0,0)

	tnet["conv5_1"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv5_1"
	transferW(tnet["conv5_1"].weight, net.params['conv5_1'][0].data.reshape(512,4608))
	transferB(tnet["conv5_1"].bias, net.params['conv5_1'][1].data)
	tnet["relu5_1"] = nn.ReLU()

	tnet["conv5_2"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv5_2"
	transferW(tnet["conv5_2"].weight, net.params['conv5_2'][0].data.reshape(512,4608))
	transferB(tnet["conv5_2"].bias, net.params['conv5_2'][1].data)
	tnet["relu5_2"] = nn.ReLU()

	tnet["conv5_3"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv5_3"
	transferW(tnet["conv5_3"].weight, net.params['conv5_3'][0].data.reshape(512,4608))
	transferB(tnet["conv5_3"].bias, net.params['conv5_3'][1].data)
	tnet["relu5_3"] = nn.ReLU()

	tnet["rpn_conv/3x3"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        if args.rpn:
           print "rpn_conv/3x3"
	   transferW(tnet["rpn_conv/3x3"].weight, net.params['rpn_conv/3x3'][0].data.reshape(512,4608))
	   transferB(tnet["rpn_conv/3x3"].bias, net.params['rpn_conv/3x3'][1].data)

	tnet["rpn_relu/3x3"] = nn.ReLU()

	tnet["rpn_cls_score"]= nn.SpatialConvolutionMM(512,18,1,1,1,1,0,0)
        if args.rpn:
            print "rpn_cls_score"
	    transferW(tnet["rpn_cls_score"].weight, net.params['rpn_cls_score'][0].data.reshape(18,512))
	    transferB(tnet["rpn_cls_score"].bias, net.params['rpn_cls_score'][1].data)

	tnet["rpn_bbox_pred"]= nn.SpatialConvolutionMM(512,36,1,1,1,1,0,0)
        if args.rpn:
            print "rpn_bbox_pred"
	    transferW(tnet["rpn_bbox_pred"].weight, net.params['rpn_bbox_pred'][0].data.reshape(36,512))
	    transferB(tnet["rpn_bbox_pred"].bias, net.params['rpn_bbox_pred'][1].data)

	tnet["fc6"] = nn.Linear(25088,4096)
        print "fc6"
	transferW(tnet["fc6"].weight, net.params['fc6'][0].data)
	transferB(tnet["fc6"].bias, net.params['fc6'][1].data)
	tnet["relu6"] = nn.ReLU()
	tnet["drop6"] = nn.Dropout(0.5)

	tnet["fc7"] = nn.Linear(4096,4096)
        print "fc7"
	transferW(tnet["fc7"].weight, net.params['fc7'][0].data)
	transferB(tnet["fc7"].bias, net.params['fc7'][1].data)
	tnet["relu7"] = nn.ReLU()
	tnet["drop7"] = nn.Dropout(0.5)

	tnet["cls_score"] = nn.Linear(4096,21)
        if args.rpn:
            print "cls_score"
	    transferW(tnet["cls_score"].weight, net.params['cls_score'][0].data)
	    transferB(tnet["cls_score"].bias, net.params['cls_score'][1].data)

	tnet["bbox_pred"] = nn.Linear(4096,84)
        if args.rpn:
            print "bbox_pred"
	    transferW(tnet["bbox_pred"].weight, net.params['bbox_pred'][0].data)
	    transferB(tnet["bbox_pred"].bias, net.params['bbox_pred'][1].data)

	print net.params
	PyTorchAug.save(args.out,tnet)
