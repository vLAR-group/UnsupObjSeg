'''
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dSprites --num_slots 7 
'''
import datetime
import time
import json
import os
from absl import app
from absl import logging
import tensorflow as tf
import numpy as np
import argparse
import multi_object_dataset as multi_object_dataset
import model as model_utils
import utils as utils

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf_config.allow_soft_placement = True
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
result_folder = 'results'
with open('../Dataset_Generation/dataset_path.json') as json_file:
    DATASET_PATH = json.load(json_file)


# We use `tf.function` compilation to speed up execution. For debugging,
# consider commenting out the `@tf.function` decorator.
@tf.function
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = utils.l2_loss(batch["image"], recon_combined)
    del recons, masks, slots  # Unused.

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

  return loss_value

def preprocess_data(batch, img_size, crop_region=None):
    image_dim = img_size
    image = batch['image']
    mask = batch['mask']

    # # to float
    image = tf.cast(image, tf.float32) 
    mask = tf.cast(mask, tf.float32) 
    image = ((image / 255.0) - 0.5) * 2.0

    # crop
    if crop_region != None:
        height_slice = slice(crop_region[0][0], crop_region[0][1])
        width_slice = slice(crop_region[1][0], crop_region[1][1])
        image = image[:, height_slice, width_slice, :]

        mask = mask[:, :, height_slice, width_slice, :]

    flat_mask, unflatten = flatten_all_but_last(mask, n_dims=3)

    # rescale
    size = tf.compat.v1.constant(
        image_dim, dtype=tf.int32, shape=[2])
    image = tf.compat.v1.image.resize_images(
        image, size, method=tf.image.ResizeMethod.BILINEAR)
    mask = tf.compat.v1.image.resize_images(
        flat_mask, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    batch['image'] = image
    batch['mask'] = mask
    return batch

def flatten_all_but_last(tensor, n_dims=1):
  shape = tensor.shape.as_list()
  batch_dims = shape[:-n_dims]
  flat_tensor = tf.reshape(tensor, [np.prod(batch_dims)] + shape[-n_dims:])

  def unflatten(other_tensor):
    other_shape = other_tensor.shape.as_list()
    return tf.reshape(other_tensor, batch_dims + other_shape[1:])

  return flat_tensor, unflatten


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, 
                    default=0, 
    )
    parser.add_argument("--batch_size", type=int, 
                        default=32, 
    )
    parser.add_argument("--num_iterations", type=int, 
                    default=3, 
    )
    parser.add_argument("--num_slots", type=int, 
                    default=7, 
    )
    parser.add_argument("--learning_rate", type=float, 
                        default=0.0004, 
    )
    parser.add_argument("--num_train_steps", type=int, 
                        default=500000, 
    )
    parser.add_argument("--warmup_steps", type=int, 
                        default=10000, 
    )
    parser.add_argument("--decay_rate", type=float, 
                        default=0.5, 
    )
    parser.add_argument("--decay_steps", type=int, 
                        default=100000, 
    )
    parser.add_argument("--log_loss_every", type=int, 
                        default=100, 
    )
    parser.add_argument("--save_ckpt_every", type=int, 
                        default=1000, 
    )
    parser.add_argument("--dataset", type=str,)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)

    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    
    if args.dataset == 'dSprites':
        dataset_path = DATASET_PATH['train']['dSprites']
    elif args.dataset == 'Tetris':
        dataset_path = DATASET_PATH['train']['Tetris']
    elif args.dataset == 'CLEVR':
        dataset_path = DATASET_PATH['train']['CLEVR']
    elif args.dataset == 'YCB':
        dataset_path = DATASET_PATH['train']['YCB']
    elif args.dataset == 'ScanNet':
        dataset_path = DATASET_PATH['train']['ScanNet']
    elif args.dataset == 'COCO':
        dataset_path = DATASET_PATH['train']['COCO']
    elif args.dataset == 'YCB_C':
        dataset_path = DATASET_PATH['train']['YCB_C']
    elif args.dataset == 'ScanNet_C':
        dataset_path = DATASET_PATH['train']['ScanNet_C']
    elif args.dataset == 'COCO_C':
        dataset_path = DATASET_PATH['train']['COCO_C']
    elif args.dataset == 'YCB_S':
        dataset_path = DATASET_PATH['train']['YCB_S']
    elif args.dataset == 'ScanNet_S':
        dataset_path = DATASET_PATH['train']['ScanNet_S']
    elif args.dataset == 'COCO_S':
        dataset_path = DATASET_PATH['train']['COCO_S']
    elif args.dataset == 'YCB_T':
        dataset_path = DATASET_PATH['train']['YCB_T']
    elif args.dataset == 'ScanNet_T':
        dataset_path = DATASET_PATH['train']['ScanNet_T']
    elif args.dataset == 'COCO_T':
        dataset_path = DATASET_PATH['train']['COCO_T']
    elif args.dataset == 'YCB_U':
        dataset_path = DATASET_PATH['train']['YCB_U']
    elif args.dataset == 'ScanNet_U':
        dataset_path = DATASET_PATH['train']['ScanNet_U']
    elif args.dataset == 'COCO_U':
        dataset_path = DATASET_PATH['train']['COCO_U']
    elif args.dataset == 'YCB_CS':
        dataset_path = DATASET_PATH['train']['YCB_CS']
    elif args.dataset == 'ScanNet_CS':
        dataset_path = DATASET_PATH['train']['ScanNet_CS']
    elif args.dataset == 'COCO_CS':
        dataset_path = DATASET_PATH['train']['COCO_CS']
    elif args.dataset == 'YCB_TU':
        dataset_path = DATASET_PATH['train']['YCB_TU']
    elif args.dataset == 'ScanNet_TU':
        dataset_path = DATASET_PATH['train']['ScanNet_TU']
    elif args.dataset == 'COCO_TU':
        dataset_path = DATASET_PATH['train']['COCO_TU']
    elif args.dataset == 'YCB_CT':
        dataset_path = DATASET_PATH['train']['YCB_CT']
    elif args.dataset == 'ScanNet_CT':
        dataset_path = DATASET_PATH['train']['ScanNet_CT']
    elif args.dataset == 'COCO_CT':
        dataset_path = DATASET_PATH['train']['COCO_CT']
    elif args.dataset == 'YCB_CU':
        dataset_path = DATASET_PATH['train']['YCB_CU']
    elif args.dataset == 'ScanNet_CU':
        dataset_path = DATASET_PATH['train']['ScanNet_CU']
    elif args.dataset == 'COCO_CU':
        dataset_path = DATASET_PATH['train']['COCO_CU']
    elif args.dataset == 'YCB_ST':
        dataset_path = DATASET_PATH['train']['YCB_ST']
    elif args.dataset == 'ScanNet_ST':
        dataset_path = DATASET_PATH['train']['ScanNet_ST']
    elif args.dataset == 'COCO_ST':
        dataset_path = DATASET_PATH['train']['COCO_ST']
    elif args.dataset == 'YCB_SU':
        dataset_path = DATASET_PATH['train']['YCB_SU']
    elif args.dataset == 'ScanNet_SU':
        dataset_path = DATASET_PATH['train']['ScanNet_SU']
    elif args.dataset == 'COCO_SU':
        dataset_path = DATASET_PATH['train']['COCO_SU']
    elif args.dataset == 'YCB_CST':
        dataset_path = DATASET_PATH['train']['YCB_CST']
    elif args.dataset == 'ScanNet_CST':
        dataset_path = DATASET_PATH['train']['ScanNet_CST']
    elif args.dataset == 'COCO_CST':
        dataset_path = DATASET_PATH['train']['COCO_CST']
    elif args.dataset == 'YCB_CSU':
        dataset_path = DATASET_PATH['train']['YCB_CSU']
    elif args.dataset == 'ScanNet_CSU':
        dataset_path = DATASET_PATH['train']['ScanNet_CSU']
    elif args.dataset == 'COCO_CSU':
        dataset_path = DATASET_PATH['train']['COCO_CSU']
    elif args.dataset == 'YCB_CTU':
        dataset_path = DATASET_PATH['train']['YCB_CTU']
    elif args.dataset == 'ScanNet_CTU':
        dataset_path = DATASET_PATH['train']['ScanNet_CTU']
    elif args.dataset == 'COCO_CTU':
        dataset_path = DATASET_PATH['train']['COCO_CTU']
    elif args.dataset == 'YCB_STU':
        dataset_path = DATASET_PATH['train']['YCB_STU']
    elif args.dataset == 'ScanNet_STU':
        dataset_path = DATASET_PATH['train']['ScanNet_STU']
    elif args.dataset == 'COCO_STU':
        dataset_path = DATASET_PATH['train']['COCO_STU']
    elif args.dataset == 'YCB_CSTU':
        dataset_path = DATASET_PATH['train']['YCB_CSTU']
    elif args.dataset == 'ScanNet_CSTU':
        dataset_path = DATASET_PATH['train']['ScanNet_CSTU']
    elif args.dataset == 'COCO_CSTU':
        dataset_path = DATASET_PATH['train']['COCO_CSTU']
    else:
        raise NotImplementedError
    

    model_dir = os.path.join(result_folder, args.dataset)
    dataset = multi_object_dataset.dataset(dataset_path)
    dataset = dataset.repeat().batch(args.batch_size, drop_remainder=True)
    dataset = dataset.shuffle(1000)
    data_iterator = dataset.make_one_shot_iterator()
    resolution = (128, 128)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'configs.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)  
    
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, epsilon=1e-08)

    model = model_utils.build_model(resolution=resolution, 
                                    batch_size=args.batch_size, 
                                    num_slots=args.num_slots,
                                    num_iterations=args.num_iterations,
                                    num_channels=3, 
                                    model_type="object_discovery")

    # Prepare checkpoint manager.
    global_step = tf.Variable(
        0, trainable=False, name="global_step", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        network=model, optimizer=optimizer, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=model_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from", ckpt_manager.latest_checkpoint)
    else:
        print('Initializing from scratch.')

    start = time.time()
    train_log_path = os.path.join(model_dir, 'train_log.json')
  
    for _ in range(args.num_train_steps):
        batch = next(data_iterator)
        # batch = clevr_preprocess_data(batch)
        batch = preprocess_data(batch, img_size=resolution, crop_region=None)

        # Learning rate warm-up.
        if global_step < args.warmup_steps:
            learning_rate = args.learning_rate * tf.cast(global_step, tf.float32) / tf.cast(args.warmup_steps, tf.float32)
        else:
            learning_rate = args.learning_rate
        learning_rate = learning_rate * (args.decay_rate ** (tf.cast(global_step, tf.float32) / tf.cast(args.decay_steps, tf.float32)))
        optimizer.lr = learning_rate.numpy()

        loss_value = train_step(batch, model, optimizer)
    
        # Update the global step. We update it before logging the loss and saving
        # the model so that the last checkpoint is saved at the last iteration.
        global_step.assign_add(1)

        # Log the training loss.
        if global_step % args.log_loss_every == 0:
            if not os.path.isfile(train_log_path): 
                train_log = {}
                train_log[int(global_step.numpy())] = float(loss_value.numpy())
            else:
                with open(train_log_path) as json_file:
                    train_log = json.load(json_file)
                    train_log[int(global_step.numpy())] = float(loss_value.numpy())
            with open(train_log_path, 'w') as f:
                json.dump(train_log, f, indent=2)
            print("Step:", global_step.numpy(),  " Loss:", loss_value.numpy(), " Time:",datetime.timedelta(seconds=time.time() - start))


        # We save the checkpoints every 1000 iterations.
        if global_step % args.save_ckpt_every == 0:
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            print("Saved checkpoint:", saved_ckpt)
