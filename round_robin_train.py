'''
Author: Samuel Remedios

Round robin training script which listens to the remote weight file, then
begins training once it's current host's turn.

During training, images must be preprocessed correctly. These images are stored
on the disk in a temporary directory until the program exits or the total
number of epochs is reached.

*** For now we will not do any preprocessing ***
If the results are poor, we will try reorientation.
Preprocessing:
    - skullstrip
    - reorientation
***                                         ***

Data must all exist in the same training directory, in the following format:
    filename_CT.nii.gz
    filename_mask.nii.gz
These images will be preprocessed and stored in a directory called tmp/ and will be
trained from there.

This occurs only if the images are not skullstripped or oriented properly.

TODO: Maybe better to require users to properly preprocess their own data?

Then on current host's turn, run the training script (either local or on server:
still have to decide) for some number of epochs.

Results are saved.


'''
import os
import sys
import numpy as np
import nibabel as nib
import utils
import TBISegmentation_for_CT_Train as ts  # Train Script
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import argparse
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from models.inception import inception

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Multi-site vs Single-site Training for TBI Segmentation with CT")
    parser.add_argument("--pattern", required=True, action="store", dest="pattern",
                        help="One of three patterns: A, AB, ABAB, respectively corresponding \
            to single site, transfer learning, and interleaved learning")
    parser.add_argument("--psize", required=False, default="45x45", action="store",
                        dest="patchsize",
                        help="Patch size, eg: 45x45. Patch sizes are separated by x and in voxels")
    parser.add_argument("--traindir", required=False, default="train", dest="traindir",
                        help="Relative path to directory of training images containing atlases \
            and masks, organized as: filename_CT.nii.gz, filename_mask.nii.gz, \
            filename2_CT.nii.gz, filename2_mask.nii.gz, etc")
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    results = parser.parse_args()

    # flags for different types of training
    single_site = False
    transfer_learning = False
    multi_site = False

    # The reasoning here behind 24 epochs is that it's divisible by both 2 and 3,
    # allowing us to do a 2-site and 3-site training experiment with even distribution
    TOTAL_EPOCHS = 50
    NUM_SITES = 2
    THIS_COMPUTER = open("host_id.cfg").read().split()[0]

    if results.pattern == "A":
        # single site trainiing
        NUM_EPOCHS = 100000000
        single_site = True
        train_style = THIS_COMPUTER
    elif results.pattern == "AB":
        # train here for 12, then elsewhere for 12
        NUM_EPOCHS = TOTAL_EPOCHS // NUM_SITES
        transfer_learning = True
        train_style = "transfer"
    elif results.pattern == "ABAB":
        # round robin training
        NUM_EPOCHS = 1
        multi_site = True
        train_style = "interleaved"
    else:
        parser.print_usage()
        sys.exit(1)

    MOUNT_POINT = os.path.join("..", "nihvandy", "ct_seg")
    WEIGHT_DIR = os.path.join(MOUNT_POINT, "weights", THIS_COMPUTER)
    TB_LOG_DIR = os.path.join(MOUNT_POINT, "tensorboard",
                              THIS_COMPUTER+"_weights_"+train_style+"_data", utils.now())
    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    if single_site:
        weight_filename = "weights_single_site" + "_" + THIS_COMPUTER + ".hdf5"
        LOGFILE = os.path.join(MOUNT_POINT, "log" +
                               "_" + THIS_COMPUTER + ".txt")
    elif transfer_learning:
        weight_filename = "weights_transfer_learning.hdf5"
        LOGFILE = os.path.join(MOUNT_POINT, "log_transfer_learning.txt")
    else:
        weight_filename = "weights_multi_site.hdf5"
        LOGFILE = os.path.join(MOUNT_POINT, "log_multisite_learning.txt")

    # files and paths
    ROUND_ROBIN_ORDER = open(os.path.join(MOUNT_POINT, "round_robin.cfg"))\
        .read().split()
    if not os.path.exists(LOGFILE):
        os.system("touch" + " " + LOGFILE)
    TRAIN_DIR = results.traindir

    PATCH_SIZE = [int(x) for x in results.patchsize.split("x")]
    BATCH_SIZE = 256

    print("*** FILE ALLOCATION COMPLETE ***")

    # load model

    # multi_site requires the weight directory be independent of the current host
    if multi_site:
        WEIGHT_DIR = os.path.join(MOUNT_POINT, "weights", "interleaved")
        if not os.path.exists(WEIGHT_DIR):
            os.makedirs(WEIGHT_DIR)

    existing_weights = os.listdir(WEIGHT_DIR)
    existing_weights.sort()
    if len(existing_weights) == 0 or single_site:
        # model = ts.GetModel2D(ds=2, numchannel=1)
        model = inception(num_channels=1, lr=1e-4)
    else:
        weight_path = os.path.join(WEIGHT_DIR, existing_weights[-1])
        model = load_model(weight_path)

    loss = "val_dice_coef"  # "tversky_loss"#"val_acc"#"dice"
    monitor = "val_dice_coef"  # "val_acc"#"val_dice_coef"

    # checkpoints
    checkpoint = ModelCheckpoint(os.path.join(WEIGHT_DIR,
                                              str(utils.now())+"_epoch_{epoch:02d}_"+loss+"_{"+monitor+":.4f}_"+weight_filename),
                                 monitor=monitor,
                                 verbose=0,)
    if multi_site:
        checkpoint = ModelCheckpoint(os.path.join(WEIGHT_DIR,
                                                  str(utils.now())+"_epoch_{epoch:02d}_"+loss+"_{"+monitor+":.4f}_host_"+THIS_COMPUTER+"_"+weight_filename),
                                     monitor=monitor,
                                     verbose=0,)

    tb = TensorBoard(log_dir=TB_LOG_DIR)
    callbacks_list = [checkpoint, tb]

    print("*** OBSERVER, MODEL, CHECKPOINTS ALLOCATED ***")

    ######### PREPROCESS TRAINING DATA #########
    DATA_DIR = os.path.join("data", "train")
    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    SKULLSTRIP_DIR = os.path.join(PREPROCESSING_DIR, "skullstripped")
    RAI_DIR = os.path.join(PREPROCESSING_DIR, "RAI")

    for d in [DATA_DIR, PREPROCESSING_DIR, SKULLSTRIP_DIR, RAI_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]

    filenames.sort()

    for filename in filenames:
        # skullstrip, then reorient CTs
        if "CT" in filename:
            utils.skullstrip(DATA_DIR, filename, SKULLSTRIP_DIR)
            utils.orient(filename, SKULLSTRIP_DIR, RAI_DIR)
        # just reorient masks and multiatlas
        elif "mask" in filename or "multiatlas" in filename:
            utils.orient(filename, DATA_DIR, RAI_DIR)

    print("*** PREPROCESSING COMPLETE ***")

    try:
        while True:

            print("*** BEGIN LOOP ***")

            # check log file to see who wrote last
            with open(LOGFILE, 'r') as f:
                logfile_data = [x.split() for x in f.readlines()]

            # if this is the first computer, then it's this computer's turn
            if len(logfile_data) > 1:
                most_recent = logfile_data[-1][1]
            else:
                most_recent = ROUND_ROBIN_ORDER[ROUND_ROBIN_ORDER.index(
                    THIS_COMPUTER)-1]

            # get current pos in round robin
            cur_pos = ROUND_ROBIN_ORDER.index(most_recent)

            # debug print statements
            if not single_site:
                print("order:", ROUND_ROBIN_ORDER)
                print("cur_pos:", cur_pos)
                print("thiscomp:", THIS_COMPUTER)
                print("calc:", ROUND_ROBIN_ORDER[(
                    cur_pos + 1) % len(ROUND_ROBIN_ORDER)])

            cur_host_turn = ROUND_ROBIN_ORDER[(
                cur_pos + 1) % len(ROUND_ROBIN_ORDER)] == THIS_COMPUTER
            if cur_host_turn or single_site:
                # TODO: figure out how to lock file properly
                # utils.lock(weight_path)

                print("*** GETTING PATCHES ***")
                ct_patches, mask_patches = ts.CreatePatchesForTraining(
                    atlasdir=RAI_DIR,
                    numatlas=len(os.listdir(RAI_DIR))//2,
                    patchsize=PATCH_SIZE,
                    max_patch=150000)

                print("Num patches:", len(ct_patches))
                print("ct_patches shape: {}\nmask_patches shape: {}".format(
                    ct_patches.shape, mask_patches.shape))

                # train for some number of epochs
                history = model.fit(ct_patches,
                                    mask_patches,
                                    batch_size=BATCH_SIZE,
                                    epochs=NUM_EPOCHS,
                                    verbose=1,
                                    validation_split=0.2,
                                    callbacks=callbacks_list,)

                '''
                utils.write_log(LOGFILE,
                                THIS_COMPUTER,
                                history.history['acc'][-1],
                                history.history['val_acc'][-1],
                                history.history['loss'][-1],)
                '''

                # utils.unlock()

            if single_site or transfer_learning:
                print("*** TRAINING COMPLETE ***")
                sys.exit(0)
            if multi_site:
                # check if all the epochs have been represented
                # if all epochs complete, then exit
                # otherwise, sleep and poll file
                if len(logfile_data) > 51:
                    print(
                        "*** MULTISITE TRAINING COMPLETE with {} epochs ***".format(len(logfile_data)))
                    sys.exit(0)
                time.sleep(120)  # sleep 120 seconds

    except KeyboardInterrupt:
        print("Keyboard interrupt")
        # observer.stop()

    # observer.join()
