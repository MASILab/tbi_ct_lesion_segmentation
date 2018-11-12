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
    results = utils.parse_args("validate")
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
    experiment_name = model_filename.split(os.sep)[-2]
    utils.save_args_to_csv(results, os.path.join("results", experiment_name))

    DATA_DIR = results.VAL_DIR

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

    print("Using model:", model_filename)

    # used only for printing result
    mean_dice = 0
    pred_vols = []
    gt_vols = []

    for filename, mask in zip(filenames, masks):
        # load nifti file data
        nii_obj = nib.load(os.path.join(PREPROCESSING_DIR, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # pad and reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        orig_shape = nii_img.shape
        nii_img = pad_image(nii_img)

        # segment
        segmented_img = apply_model_single_input(nii_img, model)
        pred_shape = segmented_img.shape

        # create nii obj
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)

        # load mask file data
        mask_obj = nib.load(os.path.join(PREPROCESSING_DIR, mask))
        mask_img = mask_obj.get_data()
        # pad the mask
        mask_img = pad_image(mask_img)
        mask_obj = nib.Nifti1Image(
            mask_img, affine=mask_obj.affine, header=mask_obj.header)

        # write statistics to file
        print("Collecting stats...")
        cur_vol_dice, cur_slices_dice, cur_vol, cur_vol_gt = utils.write_stats(filename,
                                                                               segmented_nii_obj,
                                                                               mask_obj,
                                                                               STATS_FILE,
                                                                               thresh,)

        save_slice(filename,
                   nii_img[:, :, :, 0],
                   segmented_img,
                   mask_img,
                   cur_slices_dice,
                   FIGURES_DIR)

        # crop off the padding
        diff_num_slices = int(np.abs(pred_shape[-1]-orig_shape[-1])/2)
        segmented_img = segmented_img[:, :, diff_num_slices:-diff_num_slices]

        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        utils.write_dice_scores(filename, cur_vol_dice,
                                cur_slices_dice, DICE_METRICS_FILE)

        mean_dice += cur_vol_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)

        # Reorient back to original before comparisons
        print("Reorienting...")
        utils.reorient(filename, DATA_DIR, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR, thresh)

    mean_dice = mean_dice / len(filenames)
    pred_vols = np.array(pred_vols)
    gt_vols = np.array(gt_vols)
    corr = np.corrcoef(pred_vols, gt_vols)[0, 1]
    print("*** Segmentation complete. ***")
    print("Mean DICE: {:.3f}".format(mean_dice))
    print("Volume Correlation: {:.3f}".format(corr))

    # save these two numbers to file
    metrics_path = os.path.join(STATS_DIR, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Dice: {:.4f}\nVolume Correlation: {:.4f}".format(
            mean_dice, corr))

    K.clear_session()
