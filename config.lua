----------------------------------
--CONFIG
---------------------------------

local M = { }

---------------------------------
-- SET INIT MODEL PATHS
---------------------------------
local init_model = {}
init_model.vgg16 = "init_models/vgg16.t7"
init_model.frcnn_vgg16 = "init_models/frcnn_vgg16.t7"
init_model.vgg1024 = "init_models/vgg1024.t7"
init_model.frcnn_vgg1024 = "init_models/frcnn_vgg1024.t7"

M.init_model = init_model

-------------------------------
-- COMMAND LINE OPTIONS
------------------------------

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a Faster RCNN model.')
  cmd:text()
  cmd:text('Options')

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')
  

  -- Model settings
  cmd:option('-train_remove_outbounds_boxes', 1,
    'Whether to ignore out-of-bounds boxes for sampling at training time')
  cmd:option('-anchor_type', 'voc',
    '\"densecap\", \"voc\", \"coco\"')

  -- Loss function weights
  cmd:option('-mid_box_reg_weight', 1.0,
    'Weight for box regression in the RPN')
  cmd:option('-mid_objectness_weight', 1.0,
    'Weight for box classification in the RPN')
  cmd:option('-end_box_reg_weight', 1.0,
    'Weight for box regression in the recognition network')
  cmd:option('-classification_weight',1.0, 'Weight for classification loss')
  cmd:option('-weight_decay', 5e-4, 'L2 weight decay penalty strength')
  cmd:option('-box_reg_decay', 5e-5,
    'Strength of pull that boxes experience towards their anchor')

  -- Data input settings
  cmd:option('-image_size', "^600",
    '\"720\" means longer side to be 720, \"^600\" means shorter side to be 600.')

  -- Optimization
  cmd:option('-optim', 'sgdm', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
  cmd:option('-learning_rate', 1e-3, 'learning rate to use')
  cmd:option('-optim_alpha', 0.9, 'alpha for adagrad/rmsprop/momentum/adam') --momemtum
  cmd:option('-step', 50000, 'learning rate reduction step') 
  cmd:option('-gamma', 0.1, 'reduce learning rate by 0.1') 
  cmd:option('-optim_beta', 0.999, 'beta used for adam')
  cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')
  cmd:option('-cnn_learning_rate', 1e-3, 'learning rate for the CNN')

  cmd:option('-max_iters', 60000, 'Number of iterations to run; -1 to run forever')
  cmd:option('-checkpoint_start_from', '',
    'Load model from a checkpoint instead of random initialization.')
  cmd:option('-finetune_cnn', 1,
    'Start finetuning CNN after this many iterations (-1 = never finetune)')
  cmd:option('-val_images_use', -1,
    'Number of validation images to use for evaluation; -1 to use all')

  -- Model checkpointing
  cmd:option('-save_checkpoint_every', 10000,
    'How often to save model checkpoints')
  cmd:option('-checkpoint_path', 'checkpoint.t7',
    'Name of the checkpoint file to use')

  -- Test-time model options (for evaluation)
  cmd:option('-test_rpn_nms_thresh', 0.7,
    'Test-time NMS threshold to use in the RPN')
  cmd:option('-test_final_nms_thresh', 0.3,
    'Test-time NMS threshold to use for final outputs')
  cmd:option('-test_num_proposals', 300,
    'Number of region proposal to use at test-time')

  -- Misc
  cmd:option('-id', '',
    'an id identifying this run/job; useful for cross-validation')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')
  cmd:option('-clip_final_boxes', 1,
             'Whether to clip final boxes to image boundar') --not incorprated
  cmd:option('-eval',false,
    'only evaluate')

  cmd:option('-bilinear', false, 'use. 0 for ROI pooling')
  cmd:option('-init_rpn',false , ' initialize RPN layers')
  cmd:option('-vgg1024', false, 'use smaller cnn')
  cmd:option('-vis', false, 'to visualize detections')
  cmd:text()

  local opt = cmd:parse(arg or {})
  return opt
end

return M
