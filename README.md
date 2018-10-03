# CT Hematoma Segmentation with U-net
## Background
Segment hematoma slice-wise from CT images with traumatic brain injury.

## Directions:
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

Example usage:
`python train.py --traindir data/train/ --psize 128x128 --num_patches 1000 --batch_size 128 --num_channels 1 --experiment_details my_experiment`

### Segment 

Run `segment.py` to classify a single image with some desired arguments:

{TODO}


### Test 

Run `test.py` to validate the model on some holdout data for which the ground truth is known and record metrics with some desired arguments:

`--datadir`: Path to where the unprocessed data is

`--axial_weights`: path to the trained model weights (.hdf5) to use

Example usage:
`python validate.py --datadir data/test/ --weights models/weights/my_experiment/my_weights.hdf5`

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
{TODO}
