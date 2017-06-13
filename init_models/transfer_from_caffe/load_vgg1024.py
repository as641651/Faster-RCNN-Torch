import caffe
import PyTorch
from PyTorchAug import nn
import PyTorchAug
import argparse, os


def transferW(b,a):
  print b.size()
#  print a.shape
  for i in range(b.size()[0]):
     for j in range(b.size()[1]):
        b[i][j] = a[i][j]

def transferB(b,a):
  print b.size()
#  print a.shape
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

        proto = 'vgg1024.prototxt'
        if args.rpn:
          print "Transfering RPN layers"
          proto = 'frcnn_vgg1024.prototxt'

	tnet = {}

	#net = caffe.Net('test.prototxt','/work/cv3/sankaran/faster-rcnn/data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel',caffe.TEST)
	#net = caffe.Net('test_c.prototxt','vgg_cnn_m_1024_faster_rcnn.caffemodel',caffe.TEST)
	net = caffe.Net(proto,args.model,caffe.TEST)
	print net.params['fc6'][0].data.shape
	print net.params['fc6'][1].data.shape

	tnet["conv1"] = nn.SpatialConvolutionMM(3,96,7,7,2,2,0,0)
        print "conv1"
	transferW(tnet["conv1"].weight, net.params['conv1'][0].data.reshape(96,147))
	transferB(tnet["conv1"].bias, net.params['conv1'][1].data)
	tnet["relu1"] = nn.ReLU()
	tnet["norm1"] = nn.SpatialCrossMapLRN(5,0.0005,0.75,2)
	tnet["pool1"] = nn.SpatialMaxPooling(3,3,2,2,0,0)

	tnet["conv2"] = nn.SpatialConvolutionMM(96,256,5,5,2,2,1,1)
        print "conv2"
	transferW(tnet["conv2"].weight, net.params['conv2'][0].data.reshape(256,2400))
	transferB(tnet["conv2"].bias, net.params['conv2'][1].data)
	tnet["relu2"] = nn.ReLU()
	tnet["norm2"] = nn.SpatialCrossMapLRN(5,0.0005,0.75,2)
	tnet["pool2"] = nn.SpatialMaxPooling(3,3,2,2,0,0)

	tnet["conv3"]= nn.SpatialConvolutionMM(256,512,3,3,1,1,1,1)
        print "conv3"
	transferW(tnet["conv3"].weight, net.params['conv3'][0].data.reshape(512,2304))
	transferB(tnet["conv3"].bias, net.params['conv3'][1].data)
	tnet["relu3"] = nn.ReLU()

	tnet["conv4"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv4"
	transferW(tnet["conv4"].weight, net.params['conv4'][0].data.reshape(512,4608))
	transferB(tnet["conv4"].bias, net.params['conv4'][1].data)
	tnet["relu4"] = nn.ReLU()

	tnet["conv5"]= nn.SpatialConvolutionMM(512,512,3,3,1,1,1,1)
        print "conv5"
	transferW(tnet["conv5"].weight, net.params['conv5'][0].data.reshape(512,4608))
	transferB(tnet["conv5"].bias, net.params['conv5'][1].data)
	tnet["relu5"] = nn.ReLU()


	tnet["rpn_conv/3x3"]= nn.SpatialConvolutionMM(512,256,3,3,1,1,1,1)
        if args.rpn:
            print "rpn_conv/3x3"
	    transferW(tnet["rpn_conv/3x3"].weight, net.params['rpn_conv/3x3'][0].data.reshape(256,4608))
	    transferB(tnet["rpn_conv/3x3"].bias, net.params['rpn_conv/3x3'][1].data)
	tnet["rpn_relu/3x3"] = nn.ReLU()

	tnet["rpn_cls_score"]= nn.SpatialConvolutionMM(256,18,1,1,1,1,0,0)
        if args.rpn:
            print "rpn_cls_score"
	    transferW(tnet["rpn_cls_score"].weight, net.params['rpn_cls_score'][0].data.reshape(18,256))
	    transferB(tnet["rpn_cls_score"].bias, net.params['rpn_cls_score'][1].data)

	tnet["rpn_bbox_pred"]= nn.SpatialConvolutionMM(256,36,1,1,1,1,0,0)
        if args.rpn:
            print "rpn_bbox_pred"
	    transferW(tnet["rpn_bbox_pred"].weight, net.params['rpn_bbox_pred'][0].data.reshape(36,256))
	    transferB(tnet["rpn_bbox_pred"].bias, net.params['rpn_bbox_pred'][1].data)

	tnet["fc6"] = nn.Linear(18432,4096) #512*6*6
        print "fc6"
	transferW(tnet["fc6"].weight, net.params['fc6'][0].data)
	transferB(tnet["fc6"].bias, net.params['fc6'][1].data)
	tnet["relu6"] = nn.ReLU()
	tnet["drop6"] = nn.Dropout(0.5)

	tnet["fc7"] = nn.Linear(4096,1024)
        print "fc7"
	transferW(tnet["fc7"].weight, net.params['fc7'][0].data)
	transferB(tnet["fc7"].bias, net.params['fc7'][1].data)
	tnet["relu7"] = nn.ReLU()
	tnet["drop7"] = nn.Dropout(0.5)

	tnet["cls_score"] = nn.Linear(1024,21)
        if args.rpn:
            print "cls_score"
	    transferW(tnet["cls_score"].weight, net.params['cls_score'][0].data)
	    transferB(tnet["cls_score"].bias, net.params['cls_score'][1].data)

	tnet["bbox_pred"] = nn.Linear(1024,84)
        if args.rpn:
            print "bbox_pred"
	    transferW(tnet["bbox_pred"].weight, net.params['bbox_pred'][0].data)
	    transferB(tnet["bbox_pred"].bias, net.params['bbox_pred'][1].data)

	print net.params
	PyTorchAug.save(args.out,tnet)
