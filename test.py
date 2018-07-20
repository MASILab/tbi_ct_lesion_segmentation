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
from utils import utils
from utils.save_figures import *
from utils.apply_model import apply_model, apply_model_single_input
from keras.models import load_model
from keras import backend as K
from models.losses import *

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    num_channels = results.num_channels
    model_filename = results.weights
    thresh = results.threshold
    experiment_name = model_filename.split(os.sep)[-2]
    experiment_details = os.path.basename(model_filename)[:os.path.basename(model_filename)
                                                          .find('.hdf5')]
    DATA_DIR = results.VAL_DIR

    ######################## PREPROCESS TESTING DATA ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")
    N4_SCRIPT_PATH = os.path.join("utils", "N4BiasFieldCorrection")

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results", experiment_name)
    FIGURES_DIR = os.path.join("results", experiment_name, "figures")
    SEG_DIR = os.path.join(SEG_ROOT_DIR, experiment_name)
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")

    for d in [PREPROCESSING_DIR, SEG_ROOT_DIR, STATS_DIR, SEG_DIR, REORIENT_DIR, FIGURES_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Stats file
    stat_filename = "result_" + experiment_details + ".csv"
    STATS_FILE = os.path.join(STATS_DIR, stat_filename)
    DICE_METRICS_FILE = os.path.join(
        STATS_DIR, "detailed_dice_" + experiment_details + ".csv")

    ######################## PREPROCESSING ########################
    filenames = [x for x in os.listdir(DATA_DIR)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]
    filenames.sort()

    for filename in filenames:
        final_preprocess_dir = utils.preprocess(filename,
                                                DATA_DIR,
                                                PREPROCESSING_DIR,
                                                SKULLSTRIP_SCRIPT_PATH,
                                                N4_SCRIPT_PATH)

    ######################## LOAD MODEL ########################
    model = load_model(model_filename, custom_objects=custom_losses)

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(final_preprocess_dir)
                 if not os.path.isdir(os.path.join(final_preprocess_dir, x))]
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
        nii_obj = nib.load(os.path.join(final_preprocess_dir, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))

        '''
        # TODO: experimenting with HU range
        blood_HU_range = range(3, 86)
        nii_img[np.invert(np.isin(nii_img, blood_HU_range))] = 0
        '''

        # segment
        #segmented_img = apply_model(nii_img, model)
        segmented_img = apply_model_single_input(nii_img, model)

        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        # load mask file data
        mask_obj = nib.load(os.path.join(final_preprocess_dir, mask))
        mask_img = mask_obj.get_data()

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

        utils.write_dice_scores(filename, cur_vol_dice,
                                cur_slices_dice, DICE_METRICS_FILE)

        mean_dice += cur_vol_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)

        # Reorient back to original before comparisons
        print("Reorienting...")
        utils.reorient(filename, final_preprocess_dir, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR, thresh)

    mean_dice = mean_dice / len(filenames)
    pred_vols = np.array(pred_vols)
    gt_vols = np.array(gt_vols)
    corr = np.corrcoef(pred_vols, gt_vols)[0, 1]
    print("*** Segmentation complete. ***")
    print("Mean DICE: {:.3f}".format(mean_dice))
    print("Volume Correlation:")
    print(corr)

    # save these two numbers to file
    metrics_path = os.path.join(STATS_DIR, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Dice: {:.4f}\nVolume Correlation: {:.4f}".format(
            mean_dice, corr))

K.clear_session()
