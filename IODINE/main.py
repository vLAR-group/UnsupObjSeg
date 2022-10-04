'''
CUDA_VISIBLE_DEVICES=0 python main.py -f with dSprites_train
'''
from copy import deepcopy
import os.path
import warnings
from absl import logging
import numpy as np
from sacred import Experiment, SETTINGS

# Ignore all tensorflow deprecation warnings
logging._warn_preinit_stderr = 0
warnings.filterwarnings("ignore", module=".*tensorflow.*")
import tensorflow.compat.v1 as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import sonnet as snt
from sacred.stflow import LogFileWriter
from modules import utils
import configurations

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

ex.named_config(configurations.dSprites_train)
ex.named_config(configurations.dSprites_test)
ex.named_config(configurations.Tetris_train)
ex.named_config(configurations.Tetris_test)
ex.named_config(configurations.CLEVR_train)
ex.named_config(configurations.CLEVR_test)
ex.named_config(configurations.YCB_train)
ex.named_config(configurations.YCB_test)
ex.named_config(configurations.ScanNet_train)
ex.named_config(configurations.ScanNet_test)
ex.named_config(configurations.COCO_train)
ex.named_config(configurations.COCO_test)


ex.named_config(configurations.YCB_C_train)
ex.named_config(configurations.YCB_C_test)
ex.named_config(configurations.ScanNet_C_train)
ex.named_config(configurations.ScanNet_C_test)
ex.named_config(configurations.COCO_C_train)
ex.named_config(configurations.COCO_C_test)

ex.named_config(configurations.YCB_S_train)
ex.named_config(configurations.YCB_S_test)
ex.named_config(configurations.ScanNet_S_train)
ex.named_config(configurations.ScanNet_S_test)
ex.named_config(configurations.COCO_S_train)
ex.named_config(configurations.COCO_S_test)

ex.named_config(configurations.YCB_T_train)
ex.named_config(configurations.YCB_T_test)
ex.named_config(configurations.ScanNet_T_train)
ex.named_config(configurations.ScanNet_T_test)
ex.named_config(configurations.COCO_T_train)
ex.named_config(configurations.COCO_T_test)

ex.named_config(configurations.YCB_U_train)
ex.named_config(configurations.YCB_U_test)
ex.named_config(configurations.ScanNet_U_train)
ex.named_config(configurations.ScanNet_U_test)
ex.named_config(configurations.COCO_U_train)
ex.named_config(configurations.COCO_U_test)

ex.named_config(configurations.YCB_CS_train)
ex.named_config(configurations.YCB_CS_test)
ex.named_config(configurations.ScanNet_CS_train)
ex.named_config(configurations.ScanNet_CS_test)
ex.named_config(configurations.COCO_CS_train)
ex.named_config(configurations.COCO_CS_test)

ex.named_config(configurations.YCB_TU_train)
ex.named_config(configurations.YCB_TU_test)
ex.named_config(configurations.ScanNet_TU_train)
ex.named_config(configurations.ScanNet_TU_test)
ex.named_config(configurations.COCO_TU_train)
ex.named_config(configurations.COCO_TU_test)

ex.named_config(configurations.YCB_CT_train)
ex.named_config(configurations.YCB_CT_test)
ex.named_config(configurations.ScanNet_CT_train)
ex.named_config(configurations.ScanNet_CT_test)
ex.named_config(configurations.COCO_CT_train)
ex.named_config(configurations.COCO_CT_test)

ex.named_config(configurations.YCB_CU_train)
ex.named_config(configurations.YCB_CU_test)
ex.named_config(configurations.ScanNet_CU_train)
ex.named_config(configurations.ScanNet_CU_test)
ex.named_config(configurations.COCO_CU_train)
ex.named_config(configurations.COCO_CU_test)

ex.named_config(configurations.YCB_ST_train)
ex.named_config(configurations.YCB_ST_test)
ex.named_config(configurations.ScanNet_ST_train)
ex.named_config(configurations.ScanNet_ST_test)
ex.named_config(configurations.COCO_ST_train)
ex.named_config(configurations.COCO_ST_test)

ex.named_config(configurations.YCB_SU_train)
ex.named_config(configurations.YCB_SU_test)
ex.named_config(configurations.ScanNet_SU_train)
ex.named_config(configurations.ScanNet_SU_test)
ex.named_config(configurations.COCO_SU_train)
ex.named_config(configurations.COCO_SU_test)

ex.named_config(configurations.YCB_CST_train)
ex.named_config(configurations.YCB_CST_test)
ex.named_config(configurations.ScanNet_CST_train)
ex.named_config(configurations.ScanNet_CST_test)
ex.named_config(configurations.COCO_CST_train)
ex.named_config(configurations.COCO_CST_test)

ex.named_config(configurations.YCB_CSU_train)
ex.named_config(configurations.YCB_CSU_test)
ex.named_config(configurations.ScanNet_CSU_train)
ex.named_config(configurations.ScanNet_CSU_test)
ex.named_config(configurations.COCO_CSU_train)
ex.named_config(configurations.COCO_CSU_test)

ex.named_config(configurations.YCB_CTU_train)
ex.named_config(configurations.YCB_CTU_test)
ex.named_config(configurations.ScanNet_CTU_train)
ex.named_config(configurations.ScanNet_CTU_test)
ex.named_config(configurations.COCO_CTU_train)
ex.named_config(configurations.COCO_CTU_test)

ex.named_config(configurations.YCB_STU_train)
ex.named_config(configurations.YCB_STU_test)
ex.named_config(configurations.ScanNet_STU_train)
ex.named_config(configurations.ScanNet_STU_test)
ex.named_config(configurations.COCO_STU_train)
ex.named_config(configurations.COCO_STU_test)

ex.named_config(configurations.YCB_CSTU_train)
ex.named_config(configurations.YCB_CSTU_test)
ex.named_config(configurations.ScanNet_CSTU_train)
ex.named_config(configurations.ScanNet_CSTU_test)
ex.named_config(configurations.COCO_CSTU_train)
ex.named_config(configurations.COCO_CSTU_test)
@ex.config
def default_config():
    continue_run = False  # set to continue experiment from an existing checkpoint
    # checkpoint_dir = ("iodine/checkpoints/coco_128size"
    #                  )  # if continue_run is False, "_{run_id}" will be appended
    save_summaries_steps = 10
    save_checkpoint_steps = 1000

    n_z = 64  # number of latent dimensions
    num_components = 7  # number of components (K)
    num_iters = 5

    learn_rate = 0.001
    batch_size = 4
    stop_after_steps = int(1e7)

    # Details for the dataset, model and optimizer are left empty here.
    # They can be found in the configurations for individual datasets,
    # which are provided in configurations.py and added as named configs.
    data = {}  # Dataset details will go here
    model = {}  # Model details will go here
    optimizer = {}  # Optimizer details will go here


@ex.capture
def build(identifier, _config):
    config_copy = deepcopy(_config[identifier])
    return utils.build(config_copy, identifier=identifier)


def get_train_step(model, dataset, optimizer):
    loss, scalars, _ = model(dataset("train"))
    global_step = tf.train.get_or_create_global_step()
    grads = optimizer.compute_gradients(loss)
    gradients, variables = zip(*grads)
    global_norm = tf.global_norm(gradients)
    gradients, global_norm = tf.clip_by_global_norm(
        gradients, 5.0, use_norm=global_norm)
    grads = zip(gradients, variables)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([train_op]):
        overview = model.get_overview_images(dataset("summary"))

        scalars["debug/global_grad_norm"] = global_norm

        summaries = {
            k: tf.summary.scalar(k, v) for k, v in scalars.items()
        }
        summaries.update(
            {k: tf.summary.image(k, v) for k, v in overview.items()})
        # session = tf.Session()
        # session.run(tf.initialize_all_variables())
        # with sess.as_default():

        return tf.identity(global_step), scalars, train_op


@ex.capture
def get_checkpoint_dir(continue_run, checkpoint_dir, _run, _log):
    if continue_run:
        print('resume from', checkpoint_dir)
        assert os.path.exists(checkpoint_dir)
        _log.info("Continuing run from checkpoint at {}".format(checkpoint_dir))
        return checkpoint_dir

    run_id = _run._id
    if run_id is None:  # then no observer was added that provided an _id
        if not _run.unobserved:
            _log.warning(
                "No run_id given or provided by an Observer. (Re-)using run_id=1.")
        run_id = 1
    checkpoint_dir = checkpoint_dir + "_{run_id}".format(run_id=run_id)
    _log.info(
        "Starting a new run using checkpoint dir: '{}'".format(checkpoint_dir))
    return checkpoint_dir


@ex.capture
def get_session(chkp_dir, loss, stop_after_steps, save_summaries_steps,
            save_checkpoint_steps):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    hooks = [
        tf.train.StopAtStepHook(last_step=stop_after_steps),
        tf.train.NanTensorHook(loss),
    ]

    return tf.train.MonitoredTrainingSession(
        hooks=hooks,
        config=config,
        checkpoint_dir=chkp_dir,
        save_summaries_steps=save_summaries_steps,
        save_checkpoint_steps=save_checkpoint_steps,
    )


@ex.command(unobserved=True)
def load_checkpoint(use_placeholder=False, session=None):
    dataset = build("data")
    model = build("model")
    if use_placeholder:
        inputs = dataset.get_placeholders()
    else:
        inputs = dataset()

    info = model.eval(inputs)
    if session is None:
        session = tf.Session()
    saver = tf.train.Saver()
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(session, checkpoint_file)

    print('Successfully restored Checkpoint "{}"'.format(checkpoint_file))
    # print variables
    variables = tf.global_variables() + tf.local_variables()
    for row in snt.format_variables(variables, join_lines=False):
        print(row)
    return {
        "session": session,
        "model": model,
        "info": info,
        "inputs": inputs,
        "dataset": dataset,
        'checkpoint_dir': checkpoint_dir,
    }


@ex.automain
@LogFileWriter(ex)
def main(save_summaries_steps):
    checkpoint_dir = get_checkpoint_dir()

    dataset = build("data")
    model = build("model")

    optimizer = build("optimizer")
    gstep, train_step_exports, train_op = get_train_step(model, dataset,
                                                        optimizer)
    loss, ari = [], []
    sc, hit_rate, accuracy = [], [], []
    info = model.eval(dataset("train"))

    with get_session(checkpoint_dir, train_step_exports["loss/total"]) as sess:
        while not sess.should_stop():
            out = sess.run({
                "step": gstep,
                "loss": train_step_exports["loss/total"],
                "ari": train_step_exports["loss/ari_nobg"],
                "train": train_op,
            })

            loss.append(out["loss"])
            ari.append(out["ari"])
            step = out["step"]
            if step % save_summaries_steps == 0:
                mean_loss = np.mean(loss)
                mean_ari = np.mean(ari)
                ex.log_scalar("loss", mean_loss, step)
                ex.log_scalar("ari", mean_ari, step)
                print("{step:>6d} Loss: {loss: >12.2f}\t\tARI-nobg:{ari: >6.2f}".format(
                    step=step, loss=mean_loss, ari=mean_ari))
                loss, ari = [], []
