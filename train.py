import os
import sys
import numpy as np
import nibabel as nib

from utils import utils


import TBISegmentation_for_CT_Train as ts  # Train Script
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import argparse
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from models.inception import inception, dual_pass_inception, simple_inception, phinet

if __name__ == "__main__":

    results = utils.parse_args("train")

    num_channels = results.num_channels
    num_epochs = 1000000

    WEIGHT_DIR = os.path.join("models", "weights")
    TB_LOG_DIR = os.path.join("models", "tensorboard", "weights", utils.now())

    # files and paths
    TRAIN_DIR = results.datadir

    for d in [MOUNT_POINT, WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    PATCH_SIZE = [int(x) for x in results.patchsize.split("x")]

    # load model
    model = inception(num_channels=num_channels, lr=1e-4)

    print(model.summary())

    monitor = "val_dice_coef"

    # checkpoints
    checkpoint_filename = str(utils.now()) +\
        "_epoch_{epoch:04d}_" +\
        monitor+"_{"+monitor+":.4f}_" +\
        weight_filename

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)

    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor=monitor,
                                 verbose=0,)

    tb = TensorBoard(log_dir=TB_LOG_DIR)
    callbacks_list = [checkpoint, tb]
    #TODO: add earlystopping


    ######### PREPROCESS TRAINING DATA #########
    DATA_DIR = os.path.join("data", "train")
    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")

    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]

    filenames.sort()

    for filename in filenames:
        final_preprocess_dir = utils.preprocess(filename, DATA_DIR, PREPROCESSING_DIR)

    print("*** PREPROCESSING COMPLETE ***")

    ct_patches, mask_patches = ts.CreatePatchesForTraining(
        atlasdir=RESAMPLE_DIR,
        unskullstrippeddir=RAI_ONLY_DIR,
        patchsize=PATCH_SIZE,
        max_patch=1000,  # 508257,
        num_channels=num_channels)

    print("Individual patch dimensions:", ct_patches[0].shape)
    print("Num patches:", len(ct_patches))
    print("ct_patches shape: {}\nmask_patches shape: {}".format(
        ct_patches.shape, mask_patches.shape))

    # train for some number of epochs
    history = model.fit(ct_patches,
                        mask_patches,
                        batch_size=128,
                        epochs=NUM_EPOCHS,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=callbacks_list,)
