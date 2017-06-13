require 'nn'
require 'nngraph'
require 'modules.MakeAnchors'
require 'modules.ROIPooling'
require 'modules.BilinearRoiPooling'
require 'modules.ReshapeBoxFeatures'
require 'modules.ApplyBoxTransform'
require 'modules.BoxSamplerHelper'
require 'modules.RegularizeLayer'
local config = require 'config'
local cmd = config.parse(arg)

opt = {}
opt.backend = cmd.backend
opt.box_reg_decay = cmd.box_reg_decay
opt.field_centers = {8.5,8.5,16,16} --fcnn
opt.sampler_nms_thresh = 0.7
opt.sampler_num_proposals = 2000
opt.sampler_batch_size = 128
opt.output_height = 6
opt.output_width = 6


function vgg1024_1(model)
  local net = nn.Sequential()

  net:add(model["conv1"])
  net:add(model["relu1"])
  net:add(model["norm1"])
  net:add(model["pool1"])

  if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(net, cudnn)
  end

  return net
end


function vgg1024_2(model)
  local net = nn.Sequential()
  
  net:add(model["conv2"])
  net:add(model["relu2"])
  net:add(model["norm2"])
  net:add(model["pool2"])
  
  net:add(model["conv3"])
  net:add(model["relu3"])
  net:add(model["conv4"])
  net:add(model["relu4"])
  net:add(model["conv5"])
  net:add(model["relu5"])


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
  local cat1 = nn.ConcatTable()
  cat1:add(seq)
  cat1:add(nn.ReshapeBoxFeatures(num_anchors))
  box_branch:add(cat1)

  local cat2 = nn.ConcatTable()
  cat2:add(nn.ApplyBoxTransform())
  cat2:add(nn.Identity())
  box_branch:add(cat2)

  -- Branch to produce box / not box scores for each anchor
  local rpn_branch = nn.Sequential()
  rpn_branch:add(model["rpn_cls_score"])
  rpn_branch:add(nn.ReshapeBoxFeatures(num_anchors))

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
   mod.name = 'recognition_network'
   return mod
end 
   
   

local net = {}
--local model = torch.load('imagenet_models/vgg1024.t7')
print("Loading model " .. config.init_model.vgg1024)
local model = torch.load(config.init_model.frcnn_vgg1024)
model['relu1'].inplace = true
model['relu2'].inplace = true
model['relu3'].inplace = true
model['relu4'].inplace = true
model['relu5'].inplace = true
model['relu6'].inplace = true
model['relu7'].inplace = true
model['rpn_conv/3x3'].weight:normal(0,0.0001)
model['rpn_cls_score'].weight:normal(0,0.0001)
model['rpn_bbox_pred'].weight:normal(0.0,0.0001)
model['cls_score'].weight:normal(0,1e-6)
model['bbox_pred'].weight:normal(0,1e-6)
model['rpn_conv/3x3'].bias:fill(0)
model['rpn_cls_score'].bias:fill(0)
model['rpn_bbox_pred'].bias:fill(0)
model['cls_score'].bias:fill(0)
model['bbox_pred'].bias:fill(0)

--print(model)
net.cnn_1 =  vgg1024_1(model)
net.cnn_2 =  vgg1024_2(model)
net.rpn = rpn(model)
net.sampler = nn.BoxSamplerHelper{batch_size = opt.sampler_batch_size}
net.proposal = nn.BoxSamplerHelper{batch_size = opt.sampler_batch_size,
                                   proposal = true}
if cmd.bilinear == 0 then
   print("Using ROI Pooling")
   net.pooling = nn.ROIPooling(opt.output_height, opt.output_width)
else
   print("Using Bilinear ROI Pooling")
   net.pooling = nn.BilinearRoiPooling(opt.output_height, opt.output_width)
end
net.recog = recog(model)
net.opt = opt

return net

