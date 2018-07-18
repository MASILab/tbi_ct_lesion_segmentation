import nibabel as nib
import numpy as np
import os
import sys
from PIL import Image
from .utils import remove_ext

# TODO: handle saving binary images
def save_slice(filename, ct_img_data, pred_mask_img_data, gt_mask_img_data, slices_dice, result_dst):

    best_dice = 0
    worst_dice = 1
    for idx, slice_dice in enumerate(slices_dice):
        if slice_dice != 1 and best_dice < slice_dice:
            best_dice = slice_dice
            best_slice_idx = idx

        if slice_dice != 0 and worst_dice > slice_dice:
            worst_dice = slice_dice
            worst_slice_idx = idx

    for img_data in [ct_img_data, pred_mask_img_data, gt_mask_img_data]:
        best_slice = img_data[:,:,best_slice_idx]
        best_slice = best_slice / np.max(best_slice) * 255
        best_slice = best_slice.astype(np.uint8).T
        best_im = Image.fromarray(best_slice).convert('LA')

        worst_slice = img_data[:,:,worst_slice_idx]
        worst_slice = worst_slice / np.max(worst_slice) * 255
        worst_slice = worst_slice.astype(np.uint8).T
        worst_im = Image.fromarray(worst_slice).convert('LA')

        if img_data is ct_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx) + "_orig.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx) + "_orig.png"
        elif img_data is pred_mask_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx) + "_pred.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx) + "_pred.png"
        elif img_data is gt_mask_img_data:
            best_slice_filename = remove_ext(filename) + "_best_slice_" + str(best_slice_idx) + "_gt.png"
            worst_slice_filename = remove_ext(filename) + "_worst_slice_" + str(worst_slice_idx) + "_gt.png"

        best_im.save(os.path.join(result_dst, best_slice_filename))
        worst_im.save(os.path.join(result_dst, worst_slice_filename))
