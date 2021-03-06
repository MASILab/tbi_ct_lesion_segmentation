'''
Author: Samuel W. Remedios

Segment hemorrhages from non-contrast head CT volumes slice-wise
Assumes ~0.5mm isotropic in-plane resolution
Check README.md for further details
'''
import os
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path
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
    results = utils.parse_args("test")
    num_channels = results.num_channels

    # Boolean flag to make use of gpu
    if results.GPU == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model_filename = results.weights

    ######################## FOLDER SETUP ########################
    DATA_DIR = results.segdir

    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_DIR = os.path.join(DATA_DIR, "segmentations")
    REORIENT_DIR = os.path.join(SEG_DIR, "reoriented")
    TMPDIR = os.path.join(PREPROCESSING_DIR, "tmp_intermediate_preprocessing_steps")

    ######################## LOAD MODEL ########################
    model = load_model(model_filename, custom_objects=custom_losses)
    
    ######################## GET FNAMES ########################

    infiles = [str(x) for x in Path(results.INDIR).iterdir() if ".nii" in x.suffixes]

    for infile in infiles:

        # create appropriate directories
        for d in [PREPROCESSING_DIR, SEG_DIR, REORIENT_DIR, TMPDIR]:
            if not os.path.exists(d):
                os.makedirs(d)

        ######################## PREPROCESSING ########################

        src_dir, filename = os.path.split(infile)

        preprocess.preprocess(filename,
                              src_dir=src_dir,
                              dst_dir=PREPROCESSING_DIR,
                              tmp_dir=TMPDIR,
                              verbose=0,
                              skullstrip_script_path=results.SKULLSTRIP,
                              remove_tmp_files=True)

        ######################## SEGMENT FILE ########################

        # load nifti file data
        nii_obj = nib.load(os.path.join(PREPROCESSING_DIR, filename))
        nii_img = nii_obj.get_data()

        # reshape to account for implicit "1" channel
        nii_img = np.reshape(nii_img, nii_img.shape + (1,))
        nii_img, pads = pad_image(nii_img)

        # segment
        segmented_img = apply_model_single_input(nii_img, model)

        # Crop back to original dimensions to undo the padding
        (
            (left_pad, right_pad),
            (top_pad, bottom_pad),
            _,
        ) = pads
        if left_pad != 0:
            segmented_img = segmented_img[left_pad:, ...]
        if right_pad != 0:
            segmented_img = segmented_img[:-right_pad, ...]
        if top_pad != 0:
            segmented_img = segmented_img[:, top_pad:, ...]
        if bottom_pad != 0:
            segmented_img = segmented_img[:, :-bottom_pad, ...]

        if nii_img.shape[:3] != segmented_img.shape:
            raise ValueError("Shape mismatch when cropping to original size: \n\
                    original: {}\n\
                    segmented: {}\n\
                    Please check above logic.\
                    ".format(nii_img.shape, segmented_img.shape))

        # save resultant image
        # use the header from the original image but the affine from the processed
        # the reason for this is we need the reoriented affine first to fix for 3dresample
        affine = nii_obj.affine
        header = nib.load(infile).header
        segmented_nii_obj = nib.Nifti1Image(segmented_img, affine=affine)
        segmented_filename = os.path.join(SEG_DIR, filename)
        nib.save(segmented_nii_obj, segmented_filename)

        # Reorient back to original before comparisons
        print("Reorienting...")
        utils.reorient(filename, src_dir, SEG_DIR)

        # get probability volumes and threshold image
        print("Thresholding...")
        utils.threshold(filename, REORIENT_DIR, REORIENT_DIR, 0.5)

        # move and rename file to target directory
        src_mask = os.path.join(REORIENT_DIR, filename)
        dst_mask = os.path.join(DATA_DIR, filename[:filename.find(".nii.gz")] + "_predicted_mask.nii.gz")
        shutil.move(src_mask, dst_mask)

        # finally, copy the original header and affine to the mask
        # although a dangerous operation, here we are guaranteed a match due to how
        # we generate and process the mask file
        mask_obj = nib.load(dst_mask)
        mask_img = mask_obj.get_data()
        orig_obj = nib.load(infile)
        orig_header =  orig_obj.header
        orig_affine = orig_obj.affine

        mask_obj = nib.Nifti1Image(mask_img, affine=orig_affine, header=orig_header)
        nib.save(mask_obj, dst_mask)

        # remove intermediate files and directories for this image
        if os.path.exists(TMPDIR):
            shutil.rmtree(TMPDIR)
        if os.path.exists(PREPROCESSING_DIR):
            shutil.rmtree(PREPROCESSING_DIR)
        if os.path.exists(SEG_DIR):
            shutil.rmtree(SEG_DIR)

    K.clear_session()
