'''
Author: Samuel Remedios

TODO: make this work 

'''
import os
import numpy as np
import nibabel as nib
from utils import utils
from utils.pad import pad_image

os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

def get_file_id(filename):
    f = os.path.basename(filename)
    if "mask" in f:
        return f[:f.find("_mask")]
    else:
        return f[:f.find("_thresh")]
        

if __name__ == "__main__":

    # get all mask filenames from GROUND_TRUTH_DIR
    results = utils.parse_args("calc_dice")
    GROUND_TRUTH_DIR = results.GT_DATA_DIR
    gt_ids = [get_file_id(x) for x in os.listdir(GROUND_TRUTH_DIR)]

    # interpret either individual file, or provided directory
    in_data = results.IN_DATA

    # if directory, then get all pred thresholded mask filenames
    # compute dice for files which are present in GT dir 
    if os.path.isdir(in_data):

        # used only for printing result to console
        mean_dice = 0
        pred_vols = []
        gt_vols = []

        filenames = [os.path.join(in_data, x) for x in os.listdir(in_data)
                     if not os.path.isdir(os.path.join(in_data, x))]
        # only keep the mask files
        filenames = [x for x in filenames if 'mask' in x]
        filenames.sort()


        for filename in filenames:
            # verify that the filenames are present in the ground truth directory
            if get_file_id(filename) in gt_ids:
                gt_filename = os.path.join(GROUND_TRUTH_DIR,
                                           get_file_id(filename) + "_mask.nii.gz")
                
                gt_obj = nib.load(gt_filename)
                pred_obj = nib.load(filename)

                write_stats(filename,
                            pred_obj,
                            gt_obj,
                            stats_file)

                cur_vol_dice, cur_slices_dice, cur_vol, cur_vol_gt = utils.write_stats(filename,
                                                                                       pred_obj,
                                                                                       gt_obj,
                                                                                       STATS_FILE)

                utils.write_dice_scores(filename, cur_vol_dice,
                                        cur_slices_dice, DICE_METRICS_FILE)

        mean_dice += cur_vol_dice
        pred_vols.append(cur_vol)
        gt_vols.append(cur_vol_gt)




            else:
                print("Error: {} not found in {}".format(filename, GROUND_TRUTH_DIR))

        

    # if a single file, get Dice for specific file if present in GT dir
    else:
        filename = in_data
        if get_file_id(filename) in gt_ids:
            gt_filename = os.path.join(GROUND_TRUTH_DIR,
                                       get_file_id(filename) + "_mask.nii.gz")
            
            gt_obj = nib.load(gt_filename)
            pred_obj = nib.load(filename)

            write_stats(filename,
                        pred_obj,
                        gt_obj,
                        stats_file)

        else:
            print("Error: {} not found in {}".format(filename, GROUND_TRUTH_DIR))





    for filename, mask in zip(filenames, masks):
        # load nifti file data
        nii_obj = nib.load(os.path.join(PREPROCESSING_DIR, filename))
        nii_img = nii_obj.get_data()
        header = nii_obj.header
        affine = nii_obj.affine

        # pad and reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        nii_img = pad_image(nii_img)

        # segment
        segmented_img = apply_model_single_input(nii_img, model)

        # save resultant image
        segmented_filename = os.path.join(SEG_DIR, filename)
        segmented_nii_obj = nib.Nifti1Image(
            segmented_img, affine=affine, header=header)
        nib.save(segmented_nii_obj, segmented_filename)

        # load mask file data
        mask_obj = nib.load(os.path.join(PREPROCESSING_DIR, mask))
        mask_img = mask_obj.get_data()
        mask_img = pad_image(mask_img)

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

