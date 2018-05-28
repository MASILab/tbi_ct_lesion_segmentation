from time import strftime

from .skullstrip import skullstrip
from .n4biascorrect import n4biascorrect
from .resample import resample
from .reorient import orient, reorient

import os
import argparse
import numpy as np
import nibabel as nib 
from sklearn.utils import shuffle
from skimage import measure
from subprocess import Popen, PIPE
from tqdm import tqdm
import random
import copy
import csv


def preprocess(filename, src_dir, preprocess_root_dir, skullstrip_script_path, n4_script_path):
    '''
    Preprocesses an image:
    1. skullstrip
    2. N4 bias correction
    3. resample
    4. reorient to RAI

    Params: TODO
    Returns: TODO, the directory location of the final processed image

    '''
    ########## Directory Setup ##########
    SKULLSTRIP_DIR = os.path.join(preprocess_root_dir, "skullstripped")
    N4_DIR = os.path.join(preprocess_root_dir, "n4_bias_corrected")
    RESAMPLE_DIR = os.path.join(preprocess_root_dir, "resampled")
    RAI_DIR = os.path.join(preprocess_root_dir, "RAI")

    for d in [SKULLSTRIP_DIR, N4_DIR, RESAMPLE_DIR, RAI_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    if "CT" in filename:
        skullstrip(filename, src_dir, SKULLSTRIP_DIR, skullstrip_script_path)
        n4biascorrect(filename, SKULLSTRIP_DIR, N4_DIR, n4_script_path)
        resample(filename, N4_DIR, RESAMPLE_DIR)
        orient(filename, RESAMPLE_DIR, RAI_DIR)
    elif "mask" in filename or "multiatlas" in filename:
        resample(filename, src_dir, RESAMPLE_DIR)
        orient(filename, RESAMPLE_DIR, RAI_DIR)

    return RAI_DIR




def parse_args(session):
    '''
    Parse command line arguments.

    Params:
        - session: string, one of "train", "validate", or "test"
    Returns:
        - parse_args: object, accessible representation of arguments
    '''
    parser = argparse.ArgumentParser(
        description="Arguments for Training and Testing")

    if session == "train":
        parser.add_argument('--datadir', required=True, action='store', dest='SRC_DIR',
                            help='Where the initial unprocessed data is. See readme for\
                                    further information')
        parser.add_argument('--psize', required=True, action='store', dest='patch_size',
                            help='Patch size, eg: 45x45. Patch sizes are separated by x\
                                    and in voxels')
    elif session == "test":
        parser.add_argument('--infile', required=True, action='store', dest='INFILE',
                            help='Image to classify')
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUTFILE',
                            help='Output filename (e.g. result.csv) to where the results\
                                    are written')
    elif session == "validate":
        parser.add_argument('--datadir', required=True, action='store', dest='VAL_DIR',
                            help='Where the initial unprocessed data is')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--num_channels', required=True, type=int, action='store',
                        dest='num_channels',
                        help='Number of channels to include. First is CT, second is atlas,\
                                third is unskullstripped CT')

    return parser.parse_args()



def n4biascorrect(filename, src_dir, dst_dir, script_path, verbose=0):
    '''
    N4 bias corrects a CT nifti image into data_dir/preprocessing/bias_correct_dir/

    Params:
        - filename: string, name of file to bias correct 
        - src_dir: string, path to directory where the CT to be bias corrected is
        - dst_dir: string, path to directory where the bias corrected CT is saved
        - script_path: string, path to N4 executable from ANTs
        - verbose: int, 0 for silent, 1 for verbose
    '''

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already bias corrected", filename)
        return

    if verbose == 1:
        print("N4 bias correcting", infile, "into" + " " + outfile)

    call = os.path.join(".", script_path) + " -d 3 -s 3 -c [50x50x50x50,0.0001] -i" + " " +\
        infile + " " + "-o" + " " + outfile + " -b 1 -r 1"
    os.system(call)

    if verbose == 1:
        print("Bias correction complete")


def now():
    '''
    Formats time for use in the log file
    '''
    return strftime("%Y-%m-%d_%H-%M-%S")


def write_log(log_file, host_id, acc, val_acc, loss):
    update_log_file = False
    new_log_file = False
    with open(log_file, 'r') as f:
        logfile_data = [x.split() for x in f.readlines()]
        if (len(logfile_data) >= 1 and logfile_data[-1][1] != host_id)\
                or len(logfile_data) == 0:
            update_log_file = True
        if len(logfile_data) == 0:
            new_log_file = True
    if update_log_file:
        with open(log_file, 'a') as f:
            if new_log_file:
                f.write("{:<30}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\n".format(
                    "timestamp",
                    "host_id",
                    "train_acc",
                    "val_acc",
                    "loss",))
            f.write("{:<30}\t{:<10}\t{:<10.4f}\t{:<10.4f}\t{:<10.4f}\n".format(
                    now(),
                    host_id,
                    acc,
                    val_acc,
                    loss,))


def remove_ext(filename):
    if ".nii" in filename:
        return filename[:filename.find(".nii")]
    else:
        return filename


def get_root_filename(filename):
    if "CT" in filename:
        return filename[:filename.find("_CT")]
    elif "mask" in filename:
        return filename[:filename.find("_mask")]
    else:
        return filename


def get_dice(img1, img2):
    '''
    Returns the dice score as a voxel-wise comparison of two nifti files.

    Params:
        - img1: ndarray, tensor of first .nii.gz file 
        - img2: ndarray, tensor of second .nii.gz file 
    Returns:
        - dice: float, the dice score between the two files
    '''

    empty_score = 1.0

    img_data_1 = img1.astype(np.bool)
    img_data_2 = img2.astype(np.bool)

    if img_data_1.shape != img_data_2.shape:
        print("Pred shape", img_data_1.shape)
        print("GT shape", img_data_2.shape)
        raise ValueError("Shape mismatch between files")

    img_sum = img_data_1.sum() + img_data_2.sum()
    if img_sum == 0:
        return empty_score

    intersection = np.logical_and(img_data_1, img_data_2)

    return 2. * intersection.sum() / img_sum


def write_stats(filename, gt_filename, stats_file, suffix):
    '''
    Writes to csv probability volumes and thresholded volumes.

    Params:
        - filename: string, name of segmentation NIFTI file
        - gt_filename: string, name of ground truth NIFTI file
        - stats_file: string, path and filename of .csv file to hold statistics
        - dice: float, dice score calculated when ground truth is available
        - suffix: string, _CNNLesionMembership.nii.gz as specified in Roy's seg script
    '''
    seg_filename = remove_ext(filename) + suffix
    SEVERE_HEMATOMA = 25000  # in mm^3
    threshold = 0.5

    # get ground truth severity
    nii_obj_gt = nib.load(gt_filename)
    img_data_gt = nii_obj_gt.get_data()
    zooms_gt = nii_obj_gt.header.get_zooms()
    scaling_factor_gt = zooms_gt[0] * zooms_gt[1] * zooms_gt[2]

    # get volumes
    probability_vol_gt = np.sum(img_data_gt)
    prob_thresh_vol_gt = np.sum(
        img_data_gt[np.where(img_data_gt >= threshold)])

    thresh_data_gt = img_data_gt.copy()
    thresh_data_gt[np.where(thresh_data_gt < threshold)] = 0
    thresh_data_gt[np.where(thresh_data_gt >= threshold)] = 1
    thresholded_vol_gt = np.sum(thresh_data_gt)

    thresholded_vol_mm_gt = scaling_factor_gt * thresholded_vol_gt

    # classify severity of hematoma in ground truth
    label_gt = measure.label(img_data_gt)
    props_gt = measure.regionprops(label_gt)
    if len(props_gt) > 0:
        areas = [x.area for x in props_gt]
        areas.sort()
        largest_contig_hematoma_vol_mm_gt = areas[-1] * scaling_factor_gt
    else:
        largest_contig_hematoma_vol_mm_gt = 0

    if largest_contig_hematoma_vol_mm_gt > SEVERE_HEMATOMA:
        severe_gt = 1
    else:
        severe_gt = 0

    ##### SEGMENTATION DATA #####

    # load object tensor for calculations
    nii_obj = nib.load(seg_filename)
    img_data = nii_obj.get_data()[:, :, :]
    zooms = nii_obj.header.get_zooms()
    scaling_factor = zooms[0] * zooms[1] * zooms[2]

    # get volumes
    probability_vol = np.sum(img_data)
    prob_thresh_vol = np.sum(img_data[np.where(img_data >= threshold)])

    thresh_data = img_data.copy()
    thresh_data[np.where(thresh_data < threshold)] = 0
    thresh_data[np.where(thresh_data >= threshold)] = 1
    thresholded_vol = np.sum(thresh_data)

    probability_vol_mm = scaling_factor * probability_vol
    prob_thresh_vol_mm = scaling_factor * prob_thresh_vol
    thresholded_vol_mm = scaling_factor * thresholded_vol

    # classify severity of hematoma in seg
    label = measure.label(thresh_data)
    props = measure.regionprops(label)

    if len(props) > 0:
        areas = [x.area for x in props]
        areas.sort()
        largest_contig_hematoma_vol_mm = areas[-1] * scaling_factor
    else:
        largest_contig_hematoma_vol_mm = 0

    ############## record results ############

    if largest_contig_hematoma_vol_mm > SEVERE_HEMATOMA:
        severe_pred = 1
    else:
        severe_pred = 0

    if os.path.exists(gt_filename):
        print(seg_filename, '\n', gt_filename)
        dice = get_dice(thresh_data, thresh_data_gt)
    else:
        dice = -1

    # write to file the two sums
    if not os.path.exists(stats_file):
        with open(stats_file, 'w') as csvfile:
            fieldnames = [
                "filename",
                "dice",
                "thresholded volume(mm)",
                "thresholded volume ground truth(mm)",
                "largest hematoma ground truth(mm)",
                "largest hematoma prediction(mm)",
                "severe hematoma ground truth",
                "severe hematoma pred",
                "vox dim 1(mm)",
                "vox dim 2(mm)",
                "vox dim 3(mm)",
                "probability vol(mm)",
                "probability volume(voxels)",
                "thresholded volume(voxels)",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    with open(stats_file, 'a') as csvfile:
        fieldnames = [
            "filename",
            "dice",
            "thresholded volume(mm)",
            "thresholded volume ground truth(mm)",
            "largest hematoma ground truth(mm)",
            "largest hematoma prediction(mm)",
            "severe hematoma ground truth",
            "severe hematoma pred",
            "vox dim 1(mm)",
            "vox dim 2(mm)",
            "vox dim 3(mm)",
            "probability vol(mm)",
            "probability volume(voxels)",
            "thresholded volume(voxels)",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({
                        "filename": os.path.basename(filename),
                        "dice": dice,
                        "thresholded volume(mm)": thresholded_vol_mm,
                        "thresholded volume ground truth(mm)": thresholded_vol_mm_gt,
                        "largest hematoma ground truth(mm)": largest_contig_hematoma_vol_mm_gt,
                        "largest hematoma prediction(mm)": largest_contig_hematoma_vol_mm,
                        "severe hematoma ground truth": severe_gt,
                        "severe hematoma pred": severe_pred,
                        "vox dim 1(mm)": zooms[0],
                        "vox dim 2(mm)": zooms[1],
                        "vox dim 3(mm)": zooms[2],
                        "probability vol(mm)": probability_vol_mm,
                        "probability volume(voxels)": probability_vol,
                        "thresholded volume(voxels)": thresholded_vol,
                        })

    return dice, thresholded_vol_mm, thresholded_vol_mm_gt


def threshold(filename, src_dir, dst_dir, threshold, suffix):
    '''
    Saves the thresholded image to the destination directory.
    Calls write_stats() to save statistics to file

    Params:
        - filename: string, name of segmentation NIFTI file
        - src_dir: string, source directory where segmented NIFTI file exists
        - dst_dir: string, destination directory
        - threshold: float in [0,1], threshold at which to split between 0 and 1
        - suffix: string, _CNNLesionMembership.nii.gz as specified in Roy's seg script
    '''
    seg_filename = filename
    # load object tensor for calculations
    nii_obj = nib.load(os.path.join(src_dir, seg_filename))
    img_data = nii_obj.get_data()

    # threshold image and save thresholded image
    img_data[np.where(img_data < threshold)] = 0
    img_data[np.where(img_data >= threshold)] = 1
    thresh_obj = nib.Nifti1Image(
        img_data, affine=nii_obj.affine, header=nii_obj.header)
    nib.save(thresh_obj, os.path.join(
        dst_dir, get_root_filename(seg_filename)+"_thresh.nii.gz"))


def segment(filename, src_dir, dst_dir, model_path, script_path):
    '''
    Segments a single image using Roy's neural CT segmenter
    Segmented image saved to dst_dir/
    Params:
        - filename: string, name of NIFTI file to skullstrip 
        - src_dir: string, source directory
        - dst_dir: string, destination directory
        - script_path: string, relative path to segmentation script
        - model_path: string, relative path to neural net model for Roy's script
    '''
    infile = os.path.join(src_dir, filename)

    call = ("python " + script_path
            + " --models " + model_path
            + " --ct " + infile + " --o " + dst_dir)

    os.system(call)

