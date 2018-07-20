import os
import sys
import numpy as np
import time

from utils import utils, patch_ops, logger

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam

from models.multi_gpu import ModelMGPU
from models.losses import *
from models.dual_loss_inception import inception as dual_loss_inception
from models.inception import inception

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    results = utils.parse_args("train")

    NUM_GPUS = 1

    num_channels = results.num_channels
    num_epochs = 1000000
    num_patches = results.num_patches  # 508257
    batch_size = results.batch_size
    model = results.model
    experiment_details = results.experiment_details
    loss = results.loss
    learning_rate = 1e-4

    MOUNT_POINT = os.path.join("..", "nihvandy", "ct_seg")
    LOGFILE = os.path.join(MOUNT_POINT, "multisite_training_log.txt")
    WEIGHT_DIR = os.path.join(
        MOUNT_POINT, "interleaved_weights", experiment_details)
    TB_LOG_DIR = os.path.join(MOUNT_POINT, "tensorboard", utils.now())
    THIS_COMPUTER = open("host_id.cfg").read().split()[0]

    # files and paths
    TRAIN_DIR = results.SRC_DIR

    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    PATCH_SIZE = [int(x) for x in results.patch_size.split("x")]

    # multi site ordering
    ROUND_ROBIN_ORDER = open(os.path.join(MOUNT_POINT, "round_robin.cfg"))\
        .read().split()
    if not os.path.exists(LOGFILE):
        os.system("touch" + " " + LOGFILE)

    ######### MODEL AND CALLBACKS #########

    # determine loss
    if loss == "dice_coef":
        loss = dice_coef_loss
    elif loss == "bce":
        loss = binary_crossentropy
    elif loss == "tpr":
        loss = true_positive_rate_loss
    elif loss == "cdc":
        loss = continuous_dice_coef_loss
    else:
        print("\nInvalid loss function.\n")
        sys.exit()


    monitor = "val_dice_coef"

    # checkpoints
    checkpoint_filename = str(utils.now()) + "_" +\
        monitor+"_{"+monitor+":.4f}_weights.hdf5"

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor='val_loss',
                                 save_best_only=False,
                                 mode='auto',
                                 verbose=0,)

    # tensorboard
    tb = TensorBoard(log_dir=TB_LOG_DIR)

    '''
    # early stopping
    es = EarlyStopping(monitor="val_loss",
                       min_delta=1e-4,
                       patience=10,
                       verbose=1,
                       mode='auto')
    '''

    callbacks_list = [checkpoint, tb]

    ######### PREPROCESS TRAINING DATA #########
    DATA_DIR = os.path.join("data", "train")
    HEALTHY_DIR = os.path.join(DATA_DIR, "healthy")

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    HEALTHY_PREPROCESSING_DIR = os.path.join(HEALTHY_DIR, "preprocessing")

    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")
    N4_SCRIPT_PATH = os.path.join("utils", "N4BiasFieldCorrection")

    # get unhealthy patches
    print("***** GETTING UNHEALTHY PATCHES *****")
    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]

    filenames.sort()

    for filename in filenames:
        final_preprocess_dir = utils.preprocess(filename,
                                                DATA_DIR,
                                                PREPROCESSING_DIR,
                                                SKULLSTRIP_SCRIPT_PATH,
                                                N4_SCRIPT_PATH)


    ct_patches, mask_patches = patch_ops.CreatePatchesForTraining(
        atlasdir=final_preprocess_dir,
        patchsize=PATCH_SIZE,
        max_patch=num_patches,
        num_channels=num_channels)

    print("Individual patch dimensions:", ct_patches[0].shape)
    print("Num patches:", len(ct_patches))
    print("ct_patches shape: {}\nmask_patches shape: {}".format(
        ct_patches.shape, mask_patches.shape))

    # train for some number of epochs


    # Manual early stopping
    min_delta = 1e-4
    patience = 10

    while True:
        with open(LOGFILE, 'r') as f:
            logfile_data = [x.split() for x in f.readlines()]
        # if this is the first computer, then it's this computer's turn
        if len(logfile_data) > 1:
            most_recent = logfile_data[-1][1]
            cur_patience = int(logfile_data[-1][4])
            best_loss = float(logfile_data[-1][5])
            cur_epoch = int(logfile_data[-1][6])
        else:
            most_recent = ROUND_ROBIN_ORDER[ROUND_ROBIN_ORDER.index(
                THIS_COMPUTER)-1]
            cur_patience = 0  # start with cur_patience of 0
            best_loss = 1e5  # some arbitrary large number
            cur_epoch = 0

        # get current position in round robin
        cur_pos = ROUND_ROBIN_ORDER.index(most_recent)

        # debug print statements
        print("Order:", ROUND_ROBIN_ORDER)
        print("cur_pos:", cur_pos)
        print("thiscomp:", THIS_COMPUTER)
        print("calc:", ROUND_ROBIN_ORDER[(cur_pos+1) % len(ROUND_ROBIN_ORDER)])

        cur_host_turn = ROUND_ROBIN_ORDER[(
            cur_pos+1) % len(ROUND_ROBIN_ORDER)] == THIS_COMPUTER

        if cur_host_turn:

            existing_weights = os.listdir(WEIGHT_DIR)
            existing_weights.sort()

            if len(existing_weights) == 0:
                ser_model = inception(num_channels=num_channels,
                                      loss=loss, ds=4, lr=learning_rate)
                print(ser_model.summary())
            else:
                prev_weights = os.path.join(WEIGHT_DIR, existing_weights[-1])
                print("Continuing training with", prev_weights)
                ser_model = load_model(prev_weights, custom_objects=custom_losses)


            if NUM_GPUS > 1:
                parallel_model = ModelMGPU(ser_model, NUM_GPUS)
                parallel_model.compile(Adam(lr=learning_rate),
                                       loss=loss,
                                       metrics=[dice_coef],)
                model = parallel_model

            else:
                model = ser_model

            history = model.fit(ct_patches[:100],
                                mask_patches[:100],
                                batch_size=batch_size,
                                epochs=1,
                                verbose=1,
                                validation_split=0.2,
                                callbacks=callbacks_list,)

            cur_loss = history.history['val_loss'][-1]

            # manual early stopping procedures
            if cur_loss < best_loss:
                best_loss = cur_loss
                cur_patience = 0
            elif np.abs(cur_loss - best_loss) > min_delta:
                cur_patience += 1
            cur_epoch += 1

            # write updates
            logger.write_log(LOGFILE,
                             THIS_COMPUTER,
                             history.history['val_dice_coef'][-1],
                             cur_loss,
                             cur_patience,
                             best_loss,
                             cur_epoch)

            if cur_patience >= patience:
                print("Training complete.")
                sys.exit(0)

        # else pass training to the next site
        # sleep 120 seconds; epochs will take between 4 and 22 minutes
        time.sleep(10)
