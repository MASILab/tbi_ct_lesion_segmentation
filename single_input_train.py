import os
import numpy as np

from utils import utils, patch_ops

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from models.inception import inception
from models.inception import inception

if __name__ == "__main__":

    results = utils.parse_args("train")

    num_channels = results.num_channels
    num_epochs = 1000000
    num_patches = results.num_patches
    batch_size = 256 

    WEIGHT_DIR = os.path.join("models", "weights")
    TB_LOG_DIR = os.path.join("models", "tensorboard", utils.now())

    # files and paths
    TRAIN_DIR = results.SRC_DIR

    for d in [WEIGHT_DIR, TB_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    PATCH_SIZE = [int(x) for x in results.patch_size.split("x")]

    ######### MODEL AND CALLBACKS #########
    model = inception(num_channels=num_channels, lr=1e-5)

    print(model.summary())

    monitor = "val_dice_coef"

    # checkpoints
    checkpoint_filename = str(utils.now()) +\
        "_epoch_{epoch:04d}_" +\
        monitor+"_{"+monitor+":.4f}_weights.hdf5"

    checkpoint_filename = os.path.join(WEIGHT_DIR, checkpoint_filename)
    checkpoint = ModelCheckpoint(checkpoint_filename,
                                 monitor=monitor,
                                 verbose=0,)

    # tensorboard
    tb = TensorBoard(log_dir=TB_LOG_DIR)

    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
            mode='auto', min_delta=2e-4, cooldown=5)

    # early stopping
    es = EarlyStopping(monitor="val_loss", min_delta=2e-4, patience=20,
                       verbose=1, mode='min')

    callbacks_list = [checkpoint, tb, es]

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
    history = model.fit(ct_patches,
                        mask_patches,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=callbacks_list,)
