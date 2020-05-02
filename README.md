# CT Hematoma Segmentation with U-net
## Background
Segment hematoma slice-wise from head CT images.


## Quick Alternative Start: Singularity (recommended to apply segmentations)
The included singularity image is written to process head CT volumes from a single input directory and produce corresponding mask volumes to a single output directory. Segmentations are still applied slice-wise on independent slices. The expected directory hierarchy for this singularity image is a directory with non-contrast head CT volumes of filetype `.nii.gz` in hounsfield units at around 0.5mm in-plane resolution. The best trained multi-site weights are packaged in the container.

All `.nii.gz` files in the input directory will be segmented.

To run, three arguments are needed:
```
INPUT_DIR=/path/to/input/directory
OUTPUT_DIR=/path/to/output/directory
GPU_MODE=1 # 0 means disable GPU, 1 means use all available GPUs
```
`singularity run --nv tbi_ct_lesion_segmentation.sif ${INPUT_DIR} ${OUTPUT_DIR} ${GPU_MODE}`

In order to make use of the mandatory `--nv` flag, the host computer must have a CUDA-compatible GPU with correct Nvidia Drivers and CUDA installation. Note that the `--nv` flag is required even if not setting `GPU_MODE=1`.

## Non-Singularity Directions (recommended if re-training)

### Directory Setup
Create data directories and subdirectories as below. Training will be 
executed over the data in the `train` directory and tested over data in 
the `test` directory.

The `my_images_to_segment` directory is for images we don't have the masks for

```
./tbi_ct_lesion_segmentation/
+-- data/
|   +-- train/
|   |   +-- file_1_CT.nii.gz
|   |   +-- file_1_mask.nii.gz
|   |   +-- file_2_CT.nii.gz
|   |   +-- file_2_mask.nii.gz
|   |   +-- file_3_CT.nii.gz
|   |   +-- file_3_mask.nii.gz
|   +-- test/
|   |   +-- file_1_CT.nii.gz
|   |   +-- file_1_mask.nii.gz
|   |   +-- file_2_CT.nii.gz
|   |   +-- file_2_mask.nii.gz
|   |   +-- file_3_CT.nii.gz
|   |   +-- file_3_mask.nii.gz
|   +-- my_images_to_segment/
|   |   +-- file_1.nii.gz
|   |   +-- file_2.nii.gz
|   |   +-- file_3.nii.gz
|   |   +-- file_n.nii.gz
```
### Training

Run `train.py` to train a classification model with some desired arguments.

`--datadir`: Path to where the unprocessed data is

`--psize`: Size of patches to gather, of the from `INTEGERxINTEGER`.  `INTEGER` should be a power of `2`

`--num_patches`: Number of patches to extract from each CT volume

`--batch_size`: Batch size for training the neural network

`--num_channels`: Number of channels in training images.

`--experiment_details`: A string describing this training run, for human organization

`--gpuid`: the ID of the GPU to train with, for multi-GPU systems. Enter `-1` to utilize all available GPUs

Example usage:
`python train.py --traindir data/train/ --psize 128x128 --num_patches 1000 --batch_size 128 --num_channels 1 --experiment_details my_experiment`

### Segment 

Run `segment.py` to classify a single image with some desired arguments:

`--infile`: Path to where the unprocessed single image volume is

`--inmask`: (OPTIONAL) Path to where the manually created binary mask volume is

`--weights`: path to the trained model weights (.hdf5) to use

`--segdir`: Path to directory in which segmentations will be placed.

Example usage:
`python segment.py --infile my_data/my_first_image.nii.gz --inmask mydata/my_first_image_mask.nii.gz --weights my_weighs.hdf5 --segdir my_segmentation_dir`

Example usage if manual mask is unavailable:
`python segment.py --infile my_data/my_first_image.nii.gz --weights my_weighs.hdf5 --segdir my_segmentation_dir`

### Test 

Run `test.py` to validate the model on some holdout data for which the ground truth is known and record metrics with some desired arguments:

`--datadir`: Path to where the unprocessed data is

`--weights`: path to the trained model weights (.hdf5) to use

Example usage:
`python test.py --datadir data/test/ --weights models/weights/my_experiment/my_weights.hdf5`

### Image Preprocessing
Here are all the preprocessing steps which are automatically executed in `train.py`, `validate.py`, and `test.py`.

All preprocessing code is located in `utils/utils.py`.

1) Skullstrip according to `CT_BET.sh`
2) Orient to `RAI`

These preprocessing steps require the following external programs:
- `fslmaths` (included in FSL)
- `bet2` (included in FSL)
- `3dresample` (included in AFNI)
- `3dinfo` (included in AFNI)

### References
If this code is helpful, please cite our related work:
```
@article{remedios2020distributed,
  title={Distributed deep learning across multisite datasets for generalized CT hemorrhage segmentation},
  author={Remedios, Samuel W and Roy, Snehashis and Bermudez, Camilo and Patel, Mayur B and Butman, John A and Landman, Bennett A and Pham, Dzung L},
  journal={Medical physics},
  volume={47},
  number={1},
  pages={89--98},
  year={2020},
  publisher={Wiley Online Library}
}
```
