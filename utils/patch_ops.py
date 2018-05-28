import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from tqdm import tqdm
import random
import copy
from time import strftime, time


def PadImage(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim+2*padsize
    temp = np.zeros(dim2, dtype=float)
    temp[padsize:dim[0]+padsize, padsize:dim[1] +
         padsize, padsize:dim[2]+padsize] = vol
    return temp


def get_intersection(a, b):
    '''
    Gets intersection of coordinate arrays a and b returned from np.nonzero()-style functions.
    Used to find valid centers for healthy patches.

    TODO: CURRENTLY ONLY WORKS ON 3D ARRAYS

    Params:
        - a: tuple of N np.arrays, where N is the dimension of the original image
        - b: tuple of N np.arrays, where N is the dimension of the original image
    Returns:
        - intersection: set of tuples of rank N, where N is the dimension of the original image
    '''
    # TODO: this is the slowest operation
    start_time = time()
    print(np.array(a).shape)
    print(np.array(b).shape)

    '''

    a_transpose = np.array(a).T
    b_transpose = np.array(b).T

    a_reformat = {tuple(x) for x in a_transpose}
    b_reformat = {tuple(x) for x in b_transpose}
    '''

    if len(a) == 3:
        # first change format to be a list of coordinates rather than one list per dimension
        a_reformat = [(x, y, z) for x, y, z in zip(a[0], a[1], a[2])]
        b_reformat = [(x, y, z) for x, y, z in zip(b[0], b[1], b[2])]
    elif len(a) == 2:
        a_reformat = [(x, y) for x, y in zip(a[0], a[1])]
        b_reformat = [(x, y) for x, y in zip(b[0], b[1])]
    else:  # TODO: proper error handling
        print("Shape mismatch")

    intersection = set(a_reformat) & set(b_reformat)
    print("Time taken to calculate intersection:",
          time() - start_time, "seconds")

    return intersection


def get_center_coords(ct, mask, ratio):
    '''
    Gets coordinates for center pixel of all patches.

    Params:
        - ct: 3D ndarray, image data from which to find healthy coordinates
        - mask: 3D ndarray, image data from which to find coordinates
        - ratio: float in [0,1]. 
                 If 0 or 1, skip intersection calculation for speed
    Returns:
        - healthy_coords: set of tuples of rank 3, coordinates of healthy voxels
        - lesion_coords: set of tuples of rank 3, coordinates of lesion voxels
    '''
    if ratio == 1:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        healthy_coords = set([(x, y, z) for x, y, z in zip(ct_possible_centers[0],
                                                           ct_possible_centers[1],
                                                           ct_possible_centers[2])])
        lesion_coords = {0, 0, 0}
    elif ratio == 0:
        healthy_coords = {0, 0, 0}
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # cuurently only works for 3D input images
        lesion_coords = set([(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                          lesion_coords[1],
                                                          lesion_coords[2])])
    else:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        zeros_coords = np.where(mask == 0)
        healthy_coords = get_intersection(ct_possible_centers, zeros_coords)
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # cuurently only works for 3D input images
        lesion_coords = set([(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                          lesion_coords[1],
                                                          lesion_coords[2])])

    return healthy_coords, lesion_coords


def get_patches(invols, mask, patchsize, maxpatch, num_channels, ratio):
    '''
    Gets patches from a single subject.

    Params:
        - invols: list of ndarrays, the images of interest. Each element is a channel.
                  First channel should be the CT, the other two are arbitrary extras.
        - mask: 3D ndarray, single-channel binary mask image of the lesions
        - patchsize: 2D int ndarray, two numbers corresponding to length of sides of patch 
        - maxpatch: int, maximum allowed number of patches
        - num_channels: int, number of channels to use.  Must match rank of invols.
        - ratio: float in [0,1], percentage of healthy:lesion patches w.r.t. maxpatch
                 1 == 100% healthy patches, 0.2 == 20% healthy patches, 80% lesion patches
    '''

    healthy_coords, lesion_coords = get_center_coords(invols[0], mask, ratio)

    # Using number of lesion_coords as the baseline for this decision.
    num_patches = np.minimum(maxpatch, len(lesion_coords))
    #num_patches = num_patches * (1 + ratio)

    # allocate ndarray of maxpatch size
    ct_tensor_shape = (num_patches, patchsize[0], patchsize[1], num_channels)
    mask_tensor_shape = (num_patches, patchsize[0], patchsize[1], 1)
    CTPatches = np.ndarray(ct_tensor_shape, dtype=np.float16)
    MaskPatches = np.ndarray(ct_tensor_shape, dtype=np.float16)

    # radius from the center coord to gather patches from
    patch_radius = patchsize // 2

    # set up iterators
    healthy_patch_iter = iter(healthy_coords)
    lesion_patch_iter = iter(lesion_coords)

    # order of indices in which to place patches
    # this pre-shuffles the data
    target_indices = [x for x in range(num_patches)]
    target_indices = shuffle(target_indices, random_state=0)

    # extract patches
    for counter, i in enumerate(target_indices):
        # swap over to healthy if past ratio limit
        if counter >= ratio * num_patches:
            # get coordinates from the iterator
            healthy_dims = next(healthy_patch_iter)
            x_1 = healthy_dims[0] - patch_radius[0]
            x_2 = healthy_dims[0] + patch_radius[0] + 1
            y_1 = healthy_dims[1] - patch_radius[1]
            y_2 = healthy_dims[1] + patch_radius[1] + 1
            z = healthy_dims[2]
        else:
            # get coordinates from the iterator
            lesion_dims = next(lesion_patch_iter)
            # get start and stop dimensions according to patch_radius
            x_1 = lesion_dims[0] - patch_radius[0]
            x_2 = lesion_dims[0] + patch_radius[0] + 1
            y_1 = lesion_dims[1] - patch_radius[1]
            y_2 = lesion_dims[1] + patch_radius[1] + 1
            z = lesion_dims[2]

        # place patches in ndarrays
        for c in range(num_channels):
            CTPatches[i, :, :, c] = invols[c][x_1:x_2, y_1:y_2, z]
        MaskPatches[i, :, :, 0] = mask[x_1:x_2, y_1:y_2, z]

    return CTPatches, MaskPatches


def CreatePatchesForTraining(atlasdir, patchsize, unskullstrippeddir=None,
                             max_patch=150000, num_channels=1, healthy=False):
    '''
    This code is identical to the main() function immediately below,
    except instead of training the model it returns a tuple containing
    the CT patches and the mask patches.

    As such, there is no need for the outdir parameter.

    Params:
        - TODO
        - healthy: bool, False if not going over the healthy dataset, true otherwise

    '''

    if healthy:
        # get filenames
        ct_names = os.listdir(atlasdir)
        ct_names = [x for x in ct_names if "CT" in x]
        ct_names.sort()

        numatlas = len(ct_names)

        patchsize = np.asarray(patchsize, dtype=int)
        padsize = np.max(patchsize + 1) / 2

        # hard coded for now
        healthy_ratio = 1

        total_num_patches = max_patch * numatlas
        single_subject_num_patches = total_num_patches // numatlas
        print("Total number of lesion patches =", f)
        print("Allowed total number of lesion patches =", total_num_patches)

        CT_matsize = (total_num_patches, patchsize[0], patchsize[1], num_channels)
        Mask_matsize = (total_num_patches, patchsize[0], patchsize[1], 1)
        CTPatches = np.zeros(CT_matsize, dtype=np.float16)
        MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

        count2 = 0
        count1 = 0

        indices = [x for x in range(total_num_patches)]
        indices = shuffle(indices, random_state=0)
        cur_idx = 0

        for i in tqdm(range(0, numatlas)):
            ctname = ct_names[i]
            ctname = os.path.join(atlasdir, ctname)
            temp = nib.load(ctname)
            ct = temp.get_data()
            ct = np.asarray(ct, dtype=np.float16)

            dim = ct.shape

            ctt = PadImage(ct, padsize)
            # for healthy images, the masks are all zeros
            maskt = np.zeros(shape=ctt.shape)

            invols = [ctt]

            CTPatchesA, MaskPatchesA = get_patches(invols,
                                                   maskt,
                                                   patchsize,
                                                   single_subject_num_patches,
                                                   num_channels,
                                                   ratio=healthy_ratio)

            CTPatchesA = np.asarray(CTPatchesA, dtype=np.float16)
            MaskPatchesA = np.asarray(MaskPatchesA, dtype=np.float16)

            for ct_patch, mask_patch in zip(CTPatchesA, MaskPatchesA):
                CTPatches[indices[cur_idx], :, :, :] = ct_patch
                MaskPatches[indices[cur_idx], :, :, :] = mask_patch
                cur_idx += 1

            dim = CTPatchesA.shape
            count2 = count1 + dim[0]

            count1 = count1 + dim[0]

        dim = (count2, patchsize[0], patchsize[1], int(1))

        return (CTPatches, MaskPatches)

    else:
        # get filenames
        ct_names = os.listdir(atlasdir)
        mask_names = os.listdir(atlasdir)
        multiatlas_names = os.listdir(atlasdir)
        if unskullstrippeddir:
            unskullstripped_names = os.listdir(unskullstrippeddir)

        ct_names = [x for x in ct_names if "CT" in x]
        mask_names = [x for x in mask_names if "mask" in x]
        multiatlas_names = [x for x in multiatlas_names if "multiatlas" in x]

        ct_names.sort()
        mask_names.sort()
        multiatlas_names.sort()
        if unskullstrippeddir:
            unskullstripped_names.sort()

        numatlas = len(multiatlas_names)

        patchsize = np.asarray(patchsize, dtype=int)
        padsize = np.max(patchsize + 1) / 2

        # hard coded for now
        healthy_ratio = 0

        # calculate total number of voxels for all images to pre-allocate array
        f = 0
        for i in range(0, numatlas):
            maskname = mask_names[i]
            maskname = os.path.join(atlasdir, maskname)
            temp = nib.load(maskname)
            mask = temp.get_data()
            f = f + np.sum(mask)

        print("Total number of lesion patches =", f)
        total_num_patches = int(np.minimum(max_patch * numatlas, f))
        single_subject_num_patches = total_num_patches // numatlas
        print("Allowed total number of lesion patches =", total_num_patches)

        CT_matsize = (total_num_patches, patchsize[0], patchsize[1], num_channels)
        Mask_matsize = (total_num_patches, patchsize[0], patchsize[1], 1)
        CTPatches = np.zeros(CT_matsize, dtype=np.float16)
        MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

        ID = strftime("%d-%m-%Y") + "_" + strftime("%H-%M-%S")
        #print("Unique ID is %s " % (ID))

        x = str(int(patchsize[0])) + "x" + str(int(patchsize[1]))

        count2 = 0
        count1 = 0

        indices = [x for x in range(total_num_patches)]
        indices = shuffle(indices, random_state=0)
        cur_idx = 0

        for i in tqdm(range(0, numatlas)):
            ctname = ct_names[i]
            ctname = os.path.join(atlasdir, ctname)
            #print("Reading %s" % (ctname))
            temp = nib.load(ctname)
            ct = temp.get_data()
            ct = np.asarray(ct, dtype=np.float16)

            multiatlasname = multiatlas_names[i]
            multiatlasname = os.path.join(atlasdir, multiatlasname)
            #print("Reading %s" % (multiatlasname))
            temp = nib.load(multiatlasname)
            multiatlas = temp.get_data()
            multiatlas = np.asarray(multiatlas, dtype=np.float16)

            if unskullstrippeddir:
                unskullstrippedname = unskullstripped_names[i]
                unskullstrippedname = os.path.join(atlasdir, unskullstrippedname)
                #print("Reading %s" % (unskullstrippedname))
                temp = nib.load(unskullstrippedname)
                unskullstripped = temp.get_data()
                unskullstripped = np.asarray(unskullstripped, dtype=np.float16)

            maskname = mask_names[i]
            maskname = os.path.join(atlasdir, maskname)
            #print("Reading %s" % (maskname))
            temp = nib.load(maskname)
            mask = temp.get_data()
            mask = np.asarray(mask, dtype=np.float16)

            dim = ct.shape
            #print("Image size = %d x %d x %d " % (dim[0], dim[1], dim[2]))

            ctt = PadImage(ct, padsize)
            multiatlast = PadImage(multiatlas, padsize)
            if unskullstrippeddir:
                unskullstrippedt = PadImage(unskullstripped, padsize)
            maskt = PadImage(mask, padsize)

            if unskullstrippeddir:
                invols = [ctt, multiatlast, unskullstrippedt]
            else:
                invols = [ctt, multiatlast]

            CTPatchesA, MaskPatchesA = get_patches(invols,
                                                   maskt,
                                                   patchsize,
                                                   single_subject_num_patches,
                                                   num_channels,
                                                   ratio=healthy_ratio)

            CTPatchesA = np.asarray(CTPatchesA, dtype=np.float16)
            #MultiatlasPatchesA = np.asarray(MultiatlasPatchesA, dtype=np.float16)
            MaskPatchesA = np.asarray(MaskPatchesA, dtype=np.float16)

            for ct_patch, mask_patch in zip(CTPatchesA, MaskPatchesA):
                CTPatches[indices[cur_idx], :, :, :] = ct_patch
                #CTPatches[indices[cur_idx],:,:,1] = multiatlas_patch
                MaskPatches[indices[cur_idx], :, :, :] = mask_patch
                cur_idx += 1

            dim = CTPatchesA.shape
            count2 = count1 + dim[0]

            #print("Atlas %d : indices [%d,%d]" %(i+1,count1,count2-1))
            #CTPatches[count1:count2, :, :, :] = CTPatchesA
            #MaskPatches[count1:count2, :, :, :] = MaskPatchesA
            count1 = count1 + dim[0]

        #print("Total number of patches collected = " + str(count2))
        dim = (count2, patchsize[0], patchsize[1], int(1))

        return (CTPatches, MaskPatches)
