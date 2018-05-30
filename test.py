'''
Author: Samuel Remedios

This script pipelines the following programs and functionality:
    - skullstrip optimized for CT
    - Dr. Snehashis Roy's segmentation script (which includes reorientation)
    - writes segmentation probability volumes and thresholded volumes to a csv
    - reorients the segmented brain volume, if necessary

Use of this script involves:
    - set SRC_DIR to point to the directory holding all images
    - ensure this script sits at the top level in directory, alongside data/

Input images should simply be the raw CT scans.
'''
import os
import numpy as np
import nibabel as nib
from utils import utils
from utils.apply_model import apply_model
from keras.models import load_model
from models.dual_loss_inception import inception,\
                                       true_positive_rate,\
                                       false_positive_rate,\
                                       dice_coef


if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    num_channels = results.num_channels
    model_filename = results.weights
    DATA_DIR = results.VAL_DIR

    ######################## PATH CONSTANTS ########################
    # script constants
    SEG_SUFFIX = "_CNNLesionMembership.nii.gz"

    SEG_SCRIPT_PATH = os.path.join("TBISegmentation_for_CT_Test.py")

    # SIGMOID ACTIVATION MEANS 0.5 FOR THRESH
    thresh = 0.5

    ######################## PREPROCESS TESTING DATA ########################
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")
    N4_SCRIPT_PATH = os.path.join("utils", "N4BiasFieldCorrection")

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    STATS_DIR = os.path.join("results")
    seg_dir = os.path.join(SEG_ROOT_DIR, "experiment_details_here")
    REORIENT_DIR = os.path.join(seg_dir, "reoriented")

    for d in [PREPROCESSING_DIR, SEG_ROOT_DIR, STATS_DIR, seg_dir, REORIENT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Stats file
    stat_filename = "result_" + \
        model_filename[model_filename.find("_epoch"):model_filename.find('.hdf5')] + ".csv"
    STATS_FILE = os.path.join(STATS_DIR, stat_filename)

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
    model = load_model(model_filename, custom_objects={'true_positive_rate':true_positive_rate,
                                                       'false_positive_rate':false_positive_rate,
                                                       'dice_coef':dice_coef,})

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(final_preprocess_dir)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]

    masks = [x for x in filenames if "mask" in x]
    filenames = [x for x in filenames if "CT" in x]

    filenames.sort()
    masks.sort()

    if len(filenames) != len(masks):
        print("Error, file missing. #CT:{}, #masks:{}".format(len(filenames), len(masks)))

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

        # segment
        segmented_img = apply_model(nii_img, model)

        # save resultant image
        segmented_filename = os.path.join(seg_dir, filename)
        segmented_nii_obj = nib.Nifti1Image(segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        # load mask file data
        mask_obj = nib.load(os.path.join(final_preprocess_dir, mask))
        mask_img = mask_obj.get_data()

        # write statistics to file
        print("Collecting stats...")
        cur_dice, cur_vol, cur_vol_gt = utils.write_stats(filename,
                                                          segmented_nii_obj,
                                                          mask_obj,
                                                          STATS_FILE,)

        mean_dice += cur_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)

        # Reorient back to original before comparisons
        print("Reorienting...")
        utils.reorient(filename, DATA_DIR, seg_dir)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR,)

    print("*** Segmentation complete. ***")
    print("Mean DICE: {:.3f}".format(mean_dice/len(filenames)))

    pred_vols = np.array(pred_vols)
    gt_vols = np.array(gt_vols)
    print("Volume Correlation:")
    print(np.corrcoef(pred_vols, gt_vols))

K.clear_session()
