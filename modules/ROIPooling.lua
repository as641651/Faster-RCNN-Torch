require 'torch'
require 'nn'
local box_utils = require 'utils.box_utils'

local layer,parent = torch.class('nn.ROIPooling','nn.Module')
-- This file is borrowed from https://github.com/fmassa/object-detection.torch

function layer:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}
  self.spatial_scale = 1/16.0
  self.gradInput = {torch.Tensor()}
  
end

function layer:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function layer:updateOutput(input)
--  self.pooler = {}
  collectgarbage()
  local data = input[1]
  local rois = box_utils.xcycwh_to_x1y1x2y2(input[2]):clone()
  --correct image ids if we are in parallel mode
  --if rois[1][1] ~=1 then
  --  rois[{{},{1}}] = rois[{{},{1}}] - rois[1][1]+1
  --end
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.output:resize(num_rois,s[ss-2],self.H,self.W)
  rois:add(-1):mul(self.spatial_scale):add(1):round()
  rois[{{},1}]:cmin(s[ss])
  rois[{{},2}]:cmin(s[ss-1])
  rois[{{},3}]:cmin(s[ss])
  rois[{{},4}]:cmin(s[ss-1])
  -- element access is faster if not a cuda tensor
  if rois:type() == 'torch.CudaTensor' then
    self._rois = self._rois or torch.FloatTensor()
    self._rois:resize(rois:size()):copy(rois)
    rois = self._rois
  end

  if not self._type then self._type = self.output:type() end
  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H):type(self._type))
    end
  end

  for i=1,num_rois do
    --local roi = rois[i]
    --local im_idx = roi[1]
    if rois[i][1] < 1 then rois[i][1] = 1 end
    if rois[i][3] < 1 then rois[i][3] = 1 end
    if rois[i][2] < 1 then rois[i][2] = 1 end
    if rois[i][4] < 1 then rois[i][4] = 1 end
    if rois[i][4] < rois[i][2] then rois[i][4] = rois[i][2] end 
    if rois[i][3] < rois[i][1] then rois[i][3] = rois[i][1] end 
    local im = data[{{},{rois[i][2],rois[i][4]},{rois[i][1],rois[i][3]}}]
    self.output[i] = self.pooler[i]:updateOutput(im)
  end
  return self.output
end

function layer:updateGradInput(input,gradOutput)
  collectgarbage()
  local data = input[1]
  local rois = box_utils.xcycwh_to_x1y1x2y2(input[2]):clone()
  --if rois[1][1] ~=1 then
  --  rois[{{},{1}}] = rois[{{},{1}}] - rois[1][1]+1
  --end
  if rois:type() == 'torch.CudaTensor' then
    rois = self._rois
  end
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.gradInput[1]:resizeAs(data):zero()

  for i=1,num_rois do
    --local roi = rois[i]
    --local im_idx = roi[1]
    if rois[i][1] < 1 then rois[i][1] = 1 end
    if rois[i][3] < 1 then rois[i][3] = 1 end
    if rois[i][2] < 1 then rois[i][2] = 1 end
    if rois[i][4] < 1 then rois[i][4] = 1 end
    if rois[i][4] < rois[i][2] then rois[i][4] = rois[i][2] end 
    if rois[i][3] < rois[i][1] then rois[i][3] = rois[i][1] end 
    local r = {{},{rois[i][2],rois[i][4]},{rois[i][1],rois[i][3]}}
    local im = data[r]
    local g  = self.pooler[i]:updateGradInput(im,gradOutput[i])
    self.gradInput[1][r]:add(g)
    self.pooler[i]:clearState()
  end
  return self.gradInput
end

function layer:type(type)
  parent.type(self,type)
  for i=1,#self.pooler do
    self.pooler[i]:type(type)
  end
  self._type = type
  return self
end

function layer:setImageSize(image_height, image_width)
  --dummy
  return self
end
