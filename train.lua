#! /usr/bin/env th

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'optim'
require 'nn'
require 'modules.optim_updates'
classifier = require 'classifier'
local utils = require 'utils.utils'
local eval_utils = require 'eval.eval_utils'
local cjson = require 'cjson'

--------------------------------------
--CONFIGURATION 
--------------------------------------
local config = require 'config'
local cmd = config.parse(arg)
print("Command Line opts")
print(cmd)

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------

-- Initialize training information
local opt = {}
opt.checkpoint_path = cmd.checkpoint_path
opt.max_iters = cmd.max_iters
opt.save_checkpoint_every = cmd.save_checkpoint_every
opt.gamma = cmd.gamma
opt.step = cmd.step
opt.weight_decay = cmd.weight_decay
opt.optim = cmd.optim
opt.cnn_optim = cmd.optim
opt.learning_rate = cmd.learning_rate
opt.cnn_learning_rate = cmd.cnn_learning_rate
opt.val_images_use = cmd.val_images_use
opt.optim_alpha = cmd.optim_alpha
opt.optim_beta = cmd.optim_beta
opt.optim_epsilon = cmd.optim_epsilon
opt.fine_tune_cnn = false
opt.eval = cmd.eval
opt.vis = cmd.vis
if cmd.finetune_cnn > 0 then opt.fine_tune_cnn = true end

local iter = 1
local optim_state = {}
local cnn_optim_state = {}
local results_history = {}

if classifier.checkpoint_info.path ~= nil then
   optim_state = classifier.checkpoint_info.optim_state
   cnn_optim_state = classifier.checkpoint_info.cnn_optim_state
   iter = classifier.checkpoint_info.iter
end

if opt.fine_tune_cnn then
   cnn_params, cnn_grad_params = classifier.model.cnn_2:getParameters()
end

local faknet = nn.Sequential():type(classifier.dtype)
faknet:add(classifier.model.rpn)
faknet:add(classifier.model.recog)
local params, grad_params = faknet:getParameters()

print('total number of parameters in net: ', grad_params:nElement())
if opt.fine_tune_cnn then
   print('total number of parameters in CNN: ', cnn_grad_params:nElement())
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------

local function lossFun()
  grad_params:zero()
  if opt.fine_tune_cnn then
     cnn_grad_params:zero() 
  end

  -- Fetch data using the loader
  local timer = torch.Timer()
  local info
  local data = {}
  local loading_time = utils.timeit(function()
    data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = classifier.loader:getBatch()
  end)
  print('Loading batch time:\t' .. loading_time)
  for k, v in pairs(data) do
    data[k] = v:type(classifier.dtype)
  end

  local losses, stats
  losses = {}
  losses.total_loss = 0.0
  local fb_time = utils.timeit(function()
    losses = classifier.train.forward_backward(data.image, data.gt_boxes, data.gt_labels,opt.fine_tune_cnn)
  end)
  print('Forward-backward time:\t' .. fb_time)

  -- Apply L2 regularization
  if opt.weight_decay > 0 then
    grad_params:add(opt.weight_decay, params)
    if opt.fine_tune_cnn then
       if cnn_grad_params then cnn_grad_params:add(opt.weight_decay, cnn_params) end
    end
  end

  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
while true do  

  -- Compute loss and gradient
  local losses = 0
  if not opt.eval then
     losses = lossFun()
  end
--  local losses = 0
  if iter%opt.step == 0 then 
       opt.learning_rate = opt.learning_rate*opt.gamma
       opt.cnn_learning_rate = opt.cnn_learning_rate*opt.gamma
  end
-- Parameter update
  -- perform a parameter update
  if not opt.eval then 
    if opt.optim == 'rmsprop' then
      rmsprop(params, grad_params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(params, grad_params, opt.learning_rate, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'sgd' then
      sgd(params, grad_params, opt.learning_rate)
    elseif opt.optim == 'sgdm' then
      sgdm(params, grad_params, opt.learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(params, grad_params, opt.learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'adam' then
      adam(params, grad_params, opt.learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    else
      error('bad option opt.optim')
    end

    if opt.fine_tune_cnn then
      if opt.cnn_optim == 'sgd' then
        sgd(cnn_params, cnn_grad_params, opt.cnn_learning_rate)
      elseif opt.cnn_optim == 'sgdm' then
        sgdm(cnn_params, cnn_grad_params, opt.cnn_learning_rate, opt.optim_alpha, cnn_optim_state)
      elseif opt.cnn_optim == 'sgdmom' then
        sgdmom(cnn_params, cnn_grad_params, opt.cnn_learning_rate, opt.optim_alpha, cnn_optim_state)
      elseif opt.cnn_optim == 'adam' then
        adam(cnn_params, cnn_grad_params, opt.cnn_learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, cnn_optim_state)
      else
        error('bad option for opt.cnn_optim')
      end
    end
    -- print loss and timing/benchmarks
    print(string.format('iter %d: %s', iter, utils.build_loss_string(losses)))
  else
    print("Running evaluation ....")
  end


  if (iter > 0 and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) or opt.eval then

    -- Evaluate validation performance
    local eval_kwargs = {
      model=classifier.deploy,
      loader=classifier.loader,
      split='val',
      max_images=opt.val_images_use,
      dtype=dtype,
      vis = opt.vis,
    }
    local results = eval_utils.eval_split(eval_kwargs)
    if opt.eval then break end
    results_history[iter] = results

    -- serialize a json file that has all info except the model
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.results_history = results_history
    checkpoint.iterators = classifier.loader.iterators
    cjson.encode_number_precision(4) -- number of sig digits to use in encoding
    cjson.encode_sparse_array(true, 2, 10)
    local text = cjson.encode(checkpoint)
    local file = io.open(opt.checkpoint_path .. '.json', 'w')
    file:write(text)
    file:close()
    print('wrote ' .. opt.checkpoint_path .. '.json')

    --if iter+1 == opt.max_iters  then
      -- save the optim state, for better resuming
    checkpoint.optim_state = optim_state
    checkpoint.cnn_optim_state = cnn_optim_state
    -- save the model
    checkpoint.model = classifier.model

    torch.save(opt.checkpoint_path, checkpoint)
    print('wrote ' .. opt.checkpoint_path)

    --end
  end
    
  -- stopping criterions
  iter = iter + 1
  -- Collect garbage every so often
  if iter % 33 == 0 then collectgarbage() end
--  if loss0 == nil then loss0 = losses.total_loss end
--  if losses.total_loss > loss0 * 100 then
    --print('loss seems to be exploding, quitting.')
    --break
--  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end
end
