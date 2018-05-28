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
import csv
import argparse
from skimage import measure
from subprocess import Popen, PIPE
from utils import utils
from models.inception import inception

if __name__ == "__main__":

    ######################## COMMAND LINE ARGUMENTS ########################
    results = utils.parse_args("validate")
    num_channels = results.num_channels
    model = results.weights
    DATA_DIR = results.VAL_DIR

    ######################## PATH CONSTANTS ########################
    # script constants
    SEG_SUFFIX = "_CNNLesionMembership.nii.gz"

    SEG_SCRIPT_PATH = os.path.join("TBISegmentation_for_CT_Test.py")

    # SIGMOID ACTIVATION MEANS 0.5 FOR THRESH
    thresh = 0.5

    ######################## PREPROCESS TESTING DATA ########################
    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessing")
    SEG_ROOT_DIR = os.path.join(DATA_DIR, "segmentations")
    SKULLSTRIP_SCRIPT_PATH = os.path.join("utils", "CT_BET.sh")
    N4_SCRIPT_PATH = os.path.join("utils", "N4BiasFieldCorrection")

    # Stats file
    STATS_DIR = os.path.join("results")
    if not os.path.exists(STATS_DIR):
        os.makedirs(STATS_DIR)
    stat_filename = "result_" + \
        model[model.find("_epoch"):model.find('.hdf5')] + ".csv"
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

    ######################## SEGMENT FILES ########################
    filenames = [x for x in os.listdir(final_preprocess_dir)
                 if not os.path.isdir((os.path.join(DATA_DIR, x)))]

    masks = [x for x in filenames if "mask" in x]
    multiatlases = [x for x in filenames if "multiatlas" in x]
    filenames = [x for x in filenames if "CT" in x]

    filenames.sort()
    masks.sort()
    multiatlases.sort()

    if len(filenames) != len(multiatlases) != len(masks):
        print("Error, shape mismatch", len(filenames),
              len(multiatlases), len(masks))

    # for multichannel, need to fuse results before segmentation
    imgs_to_segment = []

    if num_channels > 1:
        for filename, multiatlas in zip(filenames, multiatlases):
            print("Fusing...")
            FUSED_DIR = os.path.join(SEG_ROOT_DIR, "fused")
            if not os.path.exists(FUSED_DIR):
                os.makedirs(FUSED_DIR)

            nii_obj = nib.load(os.path.join(final_preprocess_dir, filename))
            img_shape = nii_obj.get_data().shape
            affine = nii_obj.affine
            header = nii_obj.header

            fused = np.zeros(shape=(img_shape[0],
                                    img_shape[1],
                                    img_shape[2],
                                    num_channels))

            for i in range(num_channels):
                if i == 0:
                    chan = nib.load(os.path.join(RESAMPLE_DIR, filename))
                elif i == 1:
                    chan = nib.load(os.path.join(
                        RESAMPLE_ATLAS_DIR, multiatlas))
                # elif i == 2:
                    #chan = nib.load(os.path.join(new_resample_dir_TODO, filename))

                if chan.get_data().shape == fused[:, :, :, i].shape:
                    fused[:, :, :, i] = chan.get_data()
                else:
                    # if the shapes misalign, fit the atlas into center of image
                    # the resulting mask is still the proper size
                    shape = chan.get_data().shape

                    diff = []
                    for s, f in zip(chan.get_data().shape, fused.shape[:-1]):
                        diff.append(np.abs(s-f)//2)

                    fused[:,
                          :,
                          diff[2]:shape[2]-diff[2]:,
                          i] = chan.get_data()[diff[0]:shape[0]-diff[0],
                                               diff[1]:shape[1]-diff[1],
                                               :]

            fused_obj = nib.Nifti1Image(
                fused, affine=affine, header=header)
            fused_filename = os.path.join(FUSED_DIR, filename)
            nib.save(fused_obj, fused_filename)
            imgs_to_segment.append(fused_filename)
    else:
        for filename in filenames:
            imgs_to_segment.append(filename)

    print("Using model:", model)

    # used only for printing result
    mean_dice = 0
    pred_vols = []
    gt_vols = []

    for filename, mask in zip(imgs_to_segment, masks):

        # segment
        seg_dir = os.path.join(SEG_ROOT_DIR, "experiment_details_here")

        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)
        if not os.path.exists(os.path.join(seg_dir, utils.remove_ext(filename) + SEG_SUFFIX)):
            print("Segmenting...")
            utils.segment(os.path.basename(filename),
                          final_preprocess_dir,
                          seg_dir,
                          model,
                          SEG_SCRIPT_PATH)
        else:
            print("Already segmented", filename)

        # write statistics to file
        print("Collecting stats...")
        cur_dice, cur_vol, cur_vol_gt = utils.write_stats(os.path.join(seg_dir, filename),
                                                          os.path.join(final_preprocess_dir,
                                                                       mask),
                                                          STATS_FILE,
                                                          suffix=SEG_SUFFIX)

        mean_dice += cur_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)

        # Reorient back to original before comparisons
        REORIENT_DIR = os.path.join(seg_dir, "reoriented")
        utils.reorient(filename, DATA_DIR, seg_dir, SEG_SUFFIX=SEG_SUFFIX)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR,
                        threshold=0.5, suffix=SEG_SUFFIX)

    print("*** Segmentation complete. ***")
    print("Mean DICE: {:.3f}".format(mean_dice/len(imgs_to_segment)))

    pred_vols = np.array(pred_vols)
    gt_vols = np.array(gt_vols)
    print("Volume Correlation:")
    print(np.corrcoef(pred_vols, gt_vols))
