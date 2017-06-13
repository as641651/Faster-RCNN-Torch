require 'nn'
require 'nngraph'
require 'modules.MakeAnchors'
require 'modules.ROIPooling'
require 'modules.BilinearRoiPooling'
require 'modules.ReshapeBoxFeatures'
require 'modules.ReshapeScores'
require 'modules.ReshapeBox'
require 'modules.ReshapeAnchors'
require 'modules.ApplyBoxTransform'
require 'modules.BoxSamplerHelper'
require 'modules.RegularizeLayer'
local config = require 'config'
local cmd = config.parse(arg)

opt = {}
opt.backend = cmd.backend
opt.box_reg_decay = cmd.box_reg_decay
opt.field_centers = {7.5,7.5,16,16} --fcnn

opt.rpn_low_thresh = 0.3
opt.rpn_high_thresh = 0.7
opt.rpn_batch_size = 256
opt.rpn_fg_fraction = 0.5
opt.proposal_low_thresh = 0.5
opt.proposal_high_thresh = 0.5
opt.proposal_batch_size = 128
opt.proposal_fg_fraction = 0.25

opt.output_height = 7
opt.output_width = 7
print("RPN OPTS ")
print(opt)


function vgg16_1(model)
  local net = nn.Sequential()

  net:add(model["conv1_1"])
  net:add(model["relu1_1"])
  net:add(model["conv1_2"])
  net:add(model["relu1_2"])
  net:add(model["pool1"])

  net:add(model["conv2_1"])
  net:add(model["relu2_1"])
  net:add(model["conv2_2"])
  net:add(model["relu2_2"])
  net:add(model["pool2"])

  net:add(model["conv3_1"])
  net:add(model["relu3_1"])
  net:add(model["conv3_2"])
  net:add(model["relu3_2"])
  net:add(model["conv3_3"])
  net:add(model["relu3_3"])
  net:add(model["pool3"])


  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(net, cudnn)
  end

  return net
end


function vgg16_2(model)
  local net = nn.Sequential()

  net:add(model["conv4_1"])
  net:add(model["relu4_1"])
  net:add(model["conv4_2"])
  net:add(model["relu4_2"])
  net:add(model["conv4_3"])
  net:add(model["relu4_3"])
  net:add(model["pool4"])

  net:add(model["conv5_1"])
  net:add(model["relu5_1"])
  net:add(model["conv5_2"])
  net:add(model["relu5_2"])
  net:add(model["conv5_3"])
  net:add(model["relu5_3"])


  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(net, cudnn)
  end

  return net
end


function rpn(model)
  
-- RPN returns {boxes, anchors, transforms, scores}

  anchors = torch.Tensor({
              {184, 96}, {368, 192}, {736, 384},
              {128, 128}, {256, 256}, {512, 512},
              {88, 176}, {176, 352}, {352, 704},
            }):t():clone()

  num_anchors = 9

  local rpn = nn.Sequential()
  rpn:add(model["rpn_conv/3x3"])
  rpn:add(nn.ReLU(true))
  
  -- Branch to produce box coordinates for each anchor
  -- This branch will return {boxes, {anchors, transforms}}
  local box_branch = nn.Sequential()
  box_branch:add(model["rpn_bbox_pred"])
  box_branch:add(nn.RegularizeLayer(opt.box_reg_decay))

  local x0, y0, sx, sy = unpack(opt.field_centers)
  local seq = nn.Sequential()
  seq:add(nn.MakeAnchors(x0, y0, sx, sy, anchors))
  seq:add(nn.ReshapeBoxFeatures(num_anchors))
  --seq:add(nn.ReshapeAnchors(num_anchors))
  --seq:add(nn.ReshapeBox(num_anchors))
  local cat1 = nn.ConcatTable()
  cat1:add(seq)
  cat1:add(nn.ReshapeBoxFeatures(num_anchors))
  --cat1:add(nn.ReshapeBox(num_anchors))
  box_branch:add(cat1)

  local cat2 = nn.ConcatTable()
  cat2:add(nn.ApplyBoxTransform())
  cat2:add(nn.Identity())
  box_branch:add(cat2)

  -- Branch to produce box / not box scores for each anchor
  local rpn_branch = nn.Sequential()
  rpn_branch:add(model["rpn_cls_score"])
  --rpn_branch:add(nn.ReshapeBoxFeatures(num_anchors))
  rpn_branch:add(nn.ReshapeScores(num_anchors))

  -- Concat and flatten the branches
  local concat = nn.ConcatTable()
  concat:add(box_branch)
  concat:add(rpn_branch)
  
  rpn:add(concat)
  rpn:add(nn.FlattenTable())

  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(rpn, cudnn)
  end

  return rpn
end

function recog(model)
   
   local roi_feats = nn.Identity()()

   local recog = nn:Sequential()
   recog:add(nn.View(-1):setNumInputDims(3))
   recog:add(model["fc6"])
   recog:add(model["relu6"])
   recog:add(model["drop6"])
   recog:add(model["fc7"])
   recog:add(model["relu7"])
   recog:add(model["drop7"])

   if opt.backend == 'cudnn' then
     require 'cudnn'
     cudnn.convert(recog, cudnn)
     cudnn.convert(model["cls_score"], cudnn)
     cudnn.convert(model["bbox_pred"], cudnn)
   end

   local roi_codes = recog(roi_feats)
   local class_scores = model["cls_score"](roi_codes)
   local bbox_pred = model["bbox_pred"](roi_codes)

   class_scores:annotate{name="class_scores"}
   bbox_pred:annotate{name="bbox_pred"}

   local inputs = {roi_feats}
   local outputs = {class_scores, bbox_pred}

   local mod = nn.gModule(inputs,outputs)
   mod.name = 'recognition_networ'
   return mod
end 
   
   

local net = {}
--local model = torch.load('caffe_models/vgg16_c.t7')
--local model = torch.load('caffe_models/vgg16_c_2.t7')
--local model = torch.load('caffe_models/vgg16_sp2.t7')
print("Loading model " .. config.init_model.frcnn_vgg16)
local model = torch.load(config.init_model.frcnn_vgg16)
model['relu1_1'].inplace = true
model['relu1_2'].inplace = true
model['relu2_1'].inplace = true
model['relu2_2'].inplace = true
model['relu3_1'].inplace = true
model['relu3_2'].inplace = true
model['relu3_3'].inplace = true
model['relu4_1'].inplace = true
model['relu4_2'].inplace = true
model['relu4_3'].inplace = true
model['relu5_1'].inplace = true
model['relu5_2'].inplace = true
model['relu5_3'].inplace = true
model['relu6'].inplace = true
model['relu7'].inplace = true
model['pool1']:ceil()
model['pool2']:ceil()
model['pool3']:ceil()
model['pool4']:ceil()

net.cnn_1 =  vgg16_1(model)
net.cnn_2 =  vgg16_2(model)
net.rpn = rpn(model)
print(net.cnn_1)
print(net.cnn_2)
print(net.rpn)
net.sampler = nn.BoxSamplerHelper{batch_size = opt.rpn_batch_size,
                                  low_thresh = opt.rpn_low_thresh,
                                  high_thresh = opt.rpn_high_thresh,
                                  fg_fraction = opt.rpn_fg_fraction}

net.proposal = nn.BoxSamplerHelper{batch_size = opt.proposal_batch_size,
                                  low_thresh = opt.proposal_low_thresh,
                                  high_thresh = opt.proposal_high_thresh,
                                  fg_fraction = opt.proposal_fg_fraction,
                                  proposal = true}
if not cmd.bilinear then
   print("Using ROI Pooling")
   net.pooling = nn.ROIPooling(opt.output_height, opt.output_width)
else
   print("Using Bilinear ROI Pooling")
   net.pooling = nn.BilinearRoiPooling(opt.output_height, opt.output_width)
end

net.recog = recog(model)
net.opt = opt


return net

