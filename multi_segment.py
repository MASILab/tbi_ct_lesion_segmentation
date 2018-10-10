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
import shutil
from utils import utils
from utils import preprocess
from utils.save_figures import *
from utils.apply_model import apply_model_single_input
from utils.pad import pad_image
from keras.models import load_model
from keras import backend as K
from models.losses import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("multiseg")
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

    experiment_name = model_filename.split(os.sep)[-2]
    utils.save_args_to_csv(results, os.path.join("results", experiment_name))

    ######################## FOLDER SETUP ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")

    DATA_DIR = results.DATA_DIR

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results", experiment_name)
    FIGURES_DIR = os.path.join("results", experiment_name, "figures")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_name)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")
    TMPDIR = os.path.join(
        PREPROCESSING_DIR, "tmp_intermediate_preprocessing_steps")

    for d in [PREPROCESSING_DIR,
              SEG_ROOT_DIR,
              STATS_DIR,
              SEG_DIR,
              REORIENT_DIR,
              FIGURES_DIR,
              TMPDIR]:
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

    ######################## PREPROCESSING ########################

    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    filenames.sort()

    preprocess.preprocess_dir(DATA_DIR,
                              PREPROCESSING_DIR,
                              SKULLSTRIP_SCRIPT_PATH)

    ######################## SEGMENT FILE ########################

    filenames = [x for x in os.listdir(PREPROCESSING_DIR)
                 if not os.path.isdir((os.path.join(PREPROCESSING_DIR, x)))]
    filenames.sort()

    for filename in filenames:
        # load nifti file data
        nii_obj = nib.load(os.path.join(PREPROCESSING_DIR, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))

        # segment
        segmented_img = apply_model_single_input(nii_img, model)

        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        # Reorient back to original before comparisons
        print("Reorienting...")
        utils.reorient(filename, DATA_DIR, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR, 0.5)

    if os.path.exists(TMPDIR):
        shutil.rmtree(TMPDIR)

    K.clear_session()
