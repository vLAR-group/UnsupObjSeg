
import math 
import json
with open('../Dataset_Generation/dataset_tfrecord_path.json') as json_file:
    DATASET_PATH = json.load(json_file)

def dSprites_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/dSprites"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['dSprites'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def dSprites_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/dSprites"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['dSprites'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def Tetris_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/Tetris"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['Tetris'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def Tetris_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/Tetris"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['Tetris'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def CLEVR_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/CLEVR"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['CLEVR'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def CLEVR_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/CLEVR"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['CLEVR'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_C_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_C_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_C_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_C_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_C_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_C_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_C"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_C'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_S_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_S_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_S_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_S_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_S_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_S_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_S"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_S'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_T_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_T_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_T_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_T_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_T_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_T_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_T"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_T'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_U_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_U_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_U_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_U_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_U_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_U_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_U"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_U'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CS_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CS_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CS_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CS_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CS_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CS_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CS"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CS'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_TU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_TU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_TU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_TU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_TU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_TU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_TU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_TU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CT_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CT_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CT_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CT_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CT_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CT_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CT"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CT'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_ST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_ST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_ST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_ST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_ST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_ST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_ST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_ST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_SU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_SU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_SU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_SU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_SU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_SU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_SU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_SU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CST_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CST_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CST"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CST'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CSU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CSU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CSU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CSU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CSU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CSU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CSU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CSU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_STU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_STU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_STU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_STU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_STU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_STU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_STU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_STU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CSTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['YCB_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def YCB_CSTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/YCB_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['YCB_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CSTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['ScanNet_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def ScanNet_CSTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/ScanNet_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['ScanNet_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CSTU_train():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['train']['COCO_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }

def COCO_CSTU_test():
  n_z = 64  # number of latent dimensions
  num_components = 7  # number of components (K)
  num_iters = 5
  checkpoint_dir = "checkpoints/COCO_CSTU"

  # For the paper we used 8 GPUs with a batch size of 4 each.
  # This means a total batch size of 32, which is too large for a single GPU.
  # When reducing the batch size, the learning rate should also be lowered.
  batch_size = 4
  learn_rate = 0.001 * math.sqrt(batch_size / 32)

  data = {
      "constructor": "modules.dataloader.MultiObjectDataset",
      "batch_size": batch_size,
      "path": DATASET_PATH['test']['COCO_CSTU'],
      "max_num_objects": 7,
  }

  model = {
      "constructor": "modules.iodine.IODINE",
      "n_z": n_z,
      "num_components": num_components,
      "num_iters": num_iters,
      "iter_loss_weight": "linspace",
      "coord_type": "linear",
      "decoder": {
          "constructor": "modules.decoder.ComponentDecoder",
          "pixel_decoder": {
              "constructor": "modules.networks.BroadcastConv",
              "cnn_opt": {
                  # Final channels is irrelevant with target_output_shape
                  "output_channels": [64, 64, 64, 64, None],
                  "kernel_shapes": [3],
                  "strides": [1],
                  "activation": "elu",
              },
              "coord_type": "linear",
          },
      },
      "refinement_core": {
          "constructor": "modules.refinement.RefinementCore",
          "encoder_net": {
              "constructor": "modules.networks.CNN",
              "mode": "avg_pool",
              "cnn_opt": {
                  "output_channels": [64, 64, 64, 64],
                  "strides": [2],
                  "kernel_shapes": [3],
                  "activation": "elu",
              },
              "mlp_opt": {
                  "output_sizes": [256, 256],
                  "activation": "elu"
              },
          },
          "recurrent_net": {
              "constructor": "modules.networks.LSTM",
              "hidden_sizes": [256],
          },
          "refinement_head": {
              "constructor": "modules.refinement.ResHead"
          },
      },
      "latent_dist": {
          "constructor": "modules.distributions.LocScaleDistribution",
          "dist": "normal",
          "scale_act": "softplus",
          "scale": "var",
          "name": "latent_dist",
      },
      "output_dist": {
          "constructor": "modules.distributions.MaskedMixture",
          "num_components": num_components,
          "component_dist": {
              "constructor":
                  "modules.distributions.LocScaleDistribution",
              "dist":
                  "logistic",
              "scale":
                  "fixed",
              "scale_val":
                  0.03,
              "name":
                  "pixel_distribution",
          },
      },
      "factor_evaluator": {
          
      },
  }

  optimizer = {
      "constructor": "tensorflow.train.AdamOptimizer",
      "learning_rate": {
          "constructor": "tensorflow.train.exponential_decay",
          "learning_rate": learn_rate,
          "global_step": {
              "constructor": "tensorflow.train.get_or_create_global_step"
          },
          "decay_steps": 1000000,
          "decay_rate": 0.1,
      },
      "beta1": 0.95,
  }
