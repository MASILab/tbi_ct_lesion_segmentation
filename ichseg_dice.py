import os
import nibabel
import sys
from utils.utils import *

true_dir = sys.argv[1]
true_filenames = os.listdir(true_dir)
true_filenames = [x for x in true_filenames if "_mask" in x]
true_filenames.sort()

pred_dir = sys.argv[2]
pred_filenames = os.listdir(pred_dir)
pred_filenames.sort()

mean_dice = 0
pred_vols = []
gt_vols = []

for t, p in zip(true_filenames, pred_filenames):
    true_filepath = os.path.join(true_dir, t)
    pred_filepath = os.path.join(pred_dir, p)

    true_mask = nib.load(true_filepath)
    pred_mask = nib.load(pred_filepath)

    cur_vol_dice, cur_slices_dice, cur_vol, cur_vol_gt = write_stats(
        p,
        pred_mask,
        true_mask,
        "results/reviewer_response/ichseg_results.csv"
    )

    mean_dice += cur_vol_dice
    pred_vols.append(cur_vol)
    gt_vols.append(cur_vol_gt)

mean_dice /= len(true_filenames)
pred_vols = np.array(pred_vols)
gt_vols = np.array(gt_vols)
corr = np.corrcoef(pred_vols, gt_vols)[0, 1]

with open("results/reviewer_response/ichseg_metrics.txt", 'w') as f:
    f.write("Dice: {:.4f}\nVolume Correlation: {:.4f}\n".format(
        mean_dice,
        corr))
