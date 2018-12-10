'''
Author: Samuel Remedios

Use of this script involves:
    - set SRC_DIR to point to the directory holding all images
    - ensure this script sits at the top level in directory, alongside data/

Input images should simply be the raw CT scans.

'''
import os
import numpy as np
import nibabel as nib
from subprocess import Popen, PIPE

import matplotlib.pyplot as plt

from sklearn import metrics
from utils import utils
from utils import preprocess
from utils.save_figures import *
from utils.apply_model import apply_model_single_input
from utils.pad import pad_image
from keras.models import load_model
from keras import backend as K
import keras
from models.losses import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    DATA_DIR = results.VAL_DIR
    num_channels = results.num_channels

    NUM_GPUS = 1
    if results.GPUID == None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif results.GPUID == -1:
        # find maximum number of available GPUs
        call = "nvidia-smi --list-gpus"
        pipe = Popen(call, shell=True, stdout=PIPE).stdout
        available_gpus = pipe.read().decode().splitlines()
        NUM_GPUS = len(available_gpus)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.GPUID)

    model_filename = results.weights

    thresh = results.threshold
    if DATA_DIR.split(os.sep)[1] == "test":
        dir_tag = open("host_id.cfg").read().split()[
            0] + "_" + DATA_DIR.split(os.sep)[1]
    else:
        dir_tag = DATA_DIR.split(os.sep)[1]
    experiment_name = os.path.basename(model_filename)[:os.path.basename(model_filename)
                                                       .find("_weights")] + "_" + dir_tag

    utils.save_args_to_csv(results, os.path.join("results", experiment_name))

    ######################## PREPROCESS TESTING DATA ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results", experiment_name)
    FIGURES_DIR = os.path.join("results", experiment_name, "figures")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_name)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")

    for d in [PREPROCESSING_DIR, SEG_ROOT_DIR, STATS_DIR, SEG_DIR, REORIENT_DIR, FIGURES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Stats file
    stat_filename = "result_" + experiment_name + ".csv"
    STATS_FILE = os.path.join(STATS_DIR, stat_filename)
    DICE_METRICS_FILE = os.path.join(
        STATS_DIR, "detailed_dice_" + experiment_name + ".csv")

    ######################## LOAD MODEL ########################
    model = load_model(model_filename,
                       custom_objects=custom_losses)

    ######################## NETWORK SURGERY ########################
    model.layers.pop()
    orig_output = model.layers[-1].output
    hard_output = keras.layers.Activation(
        'linear', name='raw_output')(orig_output)
    model = model.keras.models.Model(input=model.input, output=hard_output)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=[dice_coef],
                  loss=loss)

    ######################## PREPROCESSING ########################
    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    filenames.sort()

    preprocess.preprocess_dir(DATA_DIR,
                              PREPROCESSING_DIR,
                              SKULLSTRIP_SCRIPT_PATH)

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(PREPROCESSING_DIR)
                 if not os.path.isdir(os.path.join(PREPROCESSING_DIR, x))]
    masks = [x for x in filenames if "mask" in x]
    filenames = [x for x in filenames if "CT" in x]

    filenames.sort()
    masks.sort()

    if len(filenames) != len(masks):
        print("Error, file missing. #CT:{}, #masks:{}".format(
            len(filenames), len(masks)))

    # collect one histogram per image and save
    for filename, mask in zip(filenames, masks):
        # load nifti file data
        nii_obj = nib.load(os.path.join(PREPROCESSING_DIR, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # pad and reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        orig_shape = nii_img.shape

        # segment
        segmented_img = apply_model_single_input(nii_img, model)

        x = segmented_img.flatten()
        fig = plt.hist(x, bins='auto')
        plt.title("Intensities for {}".format(os.path.basename(filename)))
        hist_dir = os.path.join("stats", "raw_histograms")
        if not os.path.exists(hist_dir):
            os.path.makedirs(hist_dir)
        hist_name = os.path.join(hist_dir,
                                 "histogram_{}.png".format(os.path.basename(filename)))
        plt.savefig(fig, hist_name)

    K.clear_session()
