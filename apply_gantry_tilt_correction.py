import nibabel as nib
import pydicom
import numpy as np
import os
import math
from scipy.ndimage import affine_transform
import sys
from tqdm import tqdm

DST_DIR = os.path.join("data", "gantry_corrected")

# get all subjects with dicoms
DICOM_DIR = "DICOM"
src_dir = sys.argv[1]
subjects = [os.path.join(src_dir, x, DICOM_DIR) for x in os.listdir(src_dir)]
subjects.sort()

# get CT and mask files to transform
train_dir = os.path.join("data", "train")
test_dir = os.path.join("data", "test")

train_filenames = [os.path.join(train_dir, x) for x in os.listdir(train_dir)
                   if not os.path.isdir(os.path.join(train_dir, x))]
test_filenames = [os.path.join(test_dir, x) for x in os.listdir(test_dir)
                  if not os.path.isdir(os.path.join(test_dir, x))]

ct_filenames = train_filenames
ct_filenames.extend(test_filenames)

mask_filenames = [x for x in ct_filenames if 'mask' in x]
ct_filenames = [x for x in ct_filenames if '_CT' in x]

# sort by subject
ct_filenames.sort(key=lambda x: x.split(os.sep)[2])
mask_filenames.sort(key=lambda x: x.split(os.sep)[2])


for orig_filename, mask_filename, src_dir in tqdm(zip(ct_filenames, mask_filenames, subjects)):
    subj = src_dir.split(os.sep)[2]

    orig = nib.load(orig_filename) 
    orig_img = orig.get_data()
    mask = nib.load(mask_filename)
    mask_img = mask.get_data()

    dicom_files = os.listdir(src_dir)
    dicom_files = [x for x in dicom_files if ".dcm" in x]
    dicom_files.sort()
    sorted_dicoms = [None] * len(dicom_files)
    for filename in dicom_files:
        sorted_dicoms[pydicom.read_file(os.path.join(src_dir, filename))
                      .data_element("InstanceNumber")
                      .value
                      - 1] = filename


    # initial values
    first_dicom = pydicom.read_file(os.path.join(src_dir, sorted_dicoms[0]))
    first = first_dicom.data_element("SliceLocation").value
    delta_y = first_dicom.data_element("PixelSpacing").value[1]
    gantry_tilt = first_dicom.data_element("GantryDetectorTilt").value


    # store slice-wise transformation matrices
    transformation_matrices = []

    for i in range(len(dicom_files)):
        dicom_img = pydicom.read_file(os.path.join(src_dir, sorted_dicoms[i]))

        location = dicom_img.data_element('SliceLocation').value
        offset = math.tan(abs(gantry_tilt)*math.pi/180)*(location-first)/delta_y
        transformation_matrices.append(np.array([[1, 0, offset],
                                                 [0, 1, 0],
                                                 [0, 0, 1]]))


    # apply slice-wise transformation matrices
    new_img = []
    new_mask = []

    # padding constants
    pad_constant = -9999
    pad_amt = 50

    for i in range(len(dicom_files)):
        H = transformation_matrices[i]
        H = np.linalg.inv(H)

        # pad the slice with a value which will never occur
        ct_slice = orig_img[:,:,i]
        ct_slice = np.pad(ct_slice, pad_amt, mode='constant', constant_values=pad_constant)

        mask_slice = mask_img[:,:,i]
        mask_slice = np.pad(mask_slice, pad_amt, mode='constant', constant_values=pad_constant)


        # slight variation in how the affine transform is applied due
        # to how the VUMC data is
        # This is possibly caused by the plane orientation (eg: RAI)
        new_img.append(affine_transform(ct_slice,
                                        H[:2, :2],
                                        (H[1, 2], H[0, 2]),
                                        order=1,
                                        mode='nearest',
                                        cval=-1024))
        new_mask.append(affine_transform(mask_slice,
                                        H[:2, :2],
                                        (H[1, 2], H[0, 2]),
                                        order=1,
                                        mode='nearest',
                                        cval=-1024))

    subj_dst_dir = os.path.join(DST_DIR, subj)
    if not os.path.exists(subj_dst_dir):
        os.makedirs(subj_dst_dir)

    # crop the new CT
    # WIP code:
    #x_pad[np.ix_((x_pad > thresh).any(1),
                 #(x_pad > thresh).any(0))]

    # Save CT
    x = np.array(new_img)
    x = np.rollaxis(x, 0, 3)
    # ensure air is constant
    x[np.where(x<-1024)] = -1024

    new_nii_obj = nib.Nifti1Image(x, affine=orig.affine, header=orig.header)
    new_filename = os.path.join(subj_dst_dir, subj+"_CT.nii.gz")
    nib.save(new_nii_obj, new_filename)

    # Save mask 
    x = np.array(new_mask)
    x = np.rollaxis(x, 0, 3)
    # ensure binary mask
    # This shouldn't be a problem, but just in case
    x[np.where(x<0.5)] = 0
    x[np.where(x>=0.5)] = 1

    new_nii_obj = nib.Nifti1Image(x, affine=orig.affine, header=orig.header)
    new_filename = os.path.join(subj_dst_dir, subj+"_mask.nii.gz")
    nib.save(new_nii_obj, new_filename)
