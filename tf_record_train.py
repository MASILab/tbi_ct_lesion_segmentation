import json
import numpy as np
import os
from subprocess import Popen, PIPE
import sys

import tensorflow as tf

from utils import utils, patch_ops
from utils import preprocess

from privacy.optimizers.dp_optimizer import DPAdamOptimizer
from privacy.optimizers.gaussian_query import GaussianAverageQuery

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from models.new_losses import *
from models.new_unet import unet


tf.enable_eager_execution()

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    NUM_GPUS = 1

    num_channels = 1
    num_epochs = 1000000
    batch_size = 128
    patch_size = [128, 128, 1]
    model_architecture = "unet"
    start_time = utils.now()
    experiment_details = start_time + "_" + model_architecture + "_" +\
        "tf_record_test"
    learning_rate = 1e-4

    WEIGHT_DIR = os.path.join("models", "weights", experiment_details)
    TB_LOG_DIR = os.path.join("models", "tensorboard", start_time)

    MODEL_NAME = model_architecture + "_model_" + experiment_details

    HISTORY_PATH = os.path.join(WEIGHT_DIR, MODEL_NAME + "_history.json")

    # files and paths
    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    ######### MODEL AND CALLBACKS #########
    model = unet(
        num_channels=num_channels,
        ds=8,
        lr=learning_rate,
        verbose=1,)

    monitor = "val_dice_coef"

    # checkpoints
    checkpoint_filename = str(start_time) +\
        "_epoch_{epoch:04d}_" +\
        monitor+"_{"+monitor+":.4f}_weights.hdf5"

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto',
                                 verbose=0,)

    # early stopping
    es = EarlyStopping(monitor="val_loss",
                       min_delta=1e-4,
                       patience=10,
                       verbose=1,
                       mode='auto')

    callbacks_list = [checkpoint, es]

    ######### DATA IMPORT #########

    def parse_single_example(record):
        image_features = tf.parse_single_example(
            record,
            features={
                'dim0': tf.FixedLenFeature([], tf.int64),
                'dim1': tf.FixedLenFeature([], tf.int64),
                'dim2': tf.FixedLenFeature([], tf.int64),
                'X': tf.FixedLenFeature([], tf.string),
                'Y': tf.FixedLenFeature([], tf.string),
                'X_dtype': tf.FixedLenFeature([], tf.string),
                'Y_dtype': tf.FixedLenFeature([], tf.string),

            }
        )

        x = tf.decode_raw(image_features.get('X'), tf.float32)
        x = tf.reshape(x, patch_size)
        y = tf.decode_raw(image_features.get('Y'), tf.float32)
        y = tf.reshape(y, patch_size)

        return x, y

    TF_RECORD_FILENAME = os.path.join("data", "train", "mydata.tfrecords")
    dataset = tf.data.TFRecordDataset(TF_RECORD_FILENAME)\
        .repeat()\
        .map(parse_single_example)\
        .shuffle(buffer_size=10000)\
        .batch(batch_size)

    data_types = {
        'uint16': tf.uint16,
        'float16': tf.float16,
        'float32': tf.float32,
        'float64': tf.float64,
    }

    #num_datapoints = 0
    #for item in dataset.take(-1):
        #num_datapoints += 1
    num_datapoints = 33000

    progbar = tf.keras.utils.Progbar(target=num_datapoints)
    steps_per_epoch = num_datapoints# // batch_size

    ######### OPTIMIZER #########
    NUM_MICROBATCHES = 1
    l2_norm_clip = 1.0
    noise_multiplier = 0.0

    dp_average_query = GaussianAverageQuery(
        l2_norm_clip=l2_norm_clip,
        sum_stddev=l2_norm_clip * noise_multiplier,
        denominator=NUM_MICROBATCHES,
    )

    opt = DPAdamOptimizer(dp_average_query,
                          NUM_MICROBATCHES,
                          learning_rate=learning_rate)

    # TODO: just use DP opt
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    ######### TRAINING #########

    loss_fn = continuous_dice_coef_loss

    best_loss = 1e6
    loss_diff = 1e6
    EARLY_STOPPING_THRESHOLD = 1e-4
    EARLY_STOPPING_EPOCHS = 10
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print("\nEpoch {}".format(epoch + 1))
        progbar.update(0)

        for cur_batch, (x, y) in enumerate(dataset):

            if cur_batch * batch_size >= num_datapoints:
                break

            with tf.GradientTape(persistent=True) as gradient_tape:
                var_list = model.trainable_variables

                def cur_loss_fn():
                    #return loss_fn(model, x, y)
                    #TODO: remove mean here
                    return np.mean(loss_fn(model, x, y))

                #loss = np.mean(cur_loss_fn().numpy())
                loss = cur_loss_fn()
                dice_score = dice_coef(model, x, y).numpy()

                ''' 
                grads_and_vars = opt.compute_gradients(cur_loss_fn,
                                                       var_list,
                                                       gradient_tape=gradient_tape)
                ''' 
                grads_and_vars = opt.compute_gradients(cur_loss_fn,
                                                       var_list,)

            opt.apply_gradients(grads_and_vars)
            progbar.add(batch_size, values=[
                        ('loss', loss), ('dice', dice_score)])

        ##### END OF EPOCH CALCULATIONS #####
        loss_diff = np.abs(best_loss - loss)
        if loss < best_loss:
            checkpoint_filename = "{}_epoch_{:04d}_loss{:.4f}_weights.hdf5"\
                    .format(start_time,
                            epoch,
                            loss)
            checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
            best_loss = loss
            model.save(checkpoint_filename, overwrite=True, include_optimizer=False)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= EARLY_STOPPING_EPOCHS and\
                loss_diff >= EARLY_STOPPING_THRESHOLD:
            break

    K.clear_session()
