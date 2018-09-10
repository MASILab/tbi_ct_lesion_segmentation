import os
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from tqdm import tqdm
import random
import copy
from time import strftime, time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


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

    # These first two must be shuffled.
    if ratio == 1:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        healthy_coords = [(x, y, z) for x, y, z in zip(ct_possible_centers[0],
                                                       ct_possible_centers[1],
                                                       ct_possible_centers[2])]
        healthy_coords = set(shuffle(healthy_coords, random_state=0))
        lesion_coords = {(0, 0, 0), (0, 0, 0), (0, 0, 0), }
    elif ratio == 0:
        healthy_coords = {(0, 0, 0), (0, 0, 0), (0, 0, 0), }
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # cuurently only works for 3D input images
        lesion_coords = [(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                      lesion_coords[1],
                                                      lesion_coords[2])]
        lesion_coords = set(shuffle(lesion_coords, random_state=0))
    else:
        # ct-valid patches
        ct_possible_centers = np.nonzero(ct)
        zeros_coords = np.where(mask == 0)
        healthy_coords = get_intersection(ct_possible_centers, zeros_coords)
        # mask lesion patches
        lesion_coords = np.nonzero(mask)
        # currently only works for 3D input images
        # This does not need to be shuffled since it will be shuffled later
        lesion_coords = set([(x, y, z) for x, y, z in zip(lesion_coords[0],
                                                          lesion_coords[1],
                                                          lesion_coords[2])])

    return healthy_coords, lesion_coords


def get_patches(invols, mask, patchsize, maxpatch, num_channels):
    rng = random.SystemRandom()

    mask = np.asarray(mask, dtype=np.float32)
    patch_size = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patchsize/2).astype(dtype=int)

    # find indices of all lesions in mask volume
    mask_lesion_indices = np.nonzero(mask)
    mask_lesion_indices = np.asarray(mask_lesion_indices, dtype=int)
    total_lesion_patches = len(mask_lesion_indices[0])

    num_patches = np.minimum(maxpatch, total_lesion_patches)
    print("Number of patches used: {} out of {} (max: {})"
          .format(num_patches,
                  total_lesion_patches),
                  maxpatch)

    randidx=rng.sample(range(0, total_lesion_patches), num_patches)
    # here, 3 corresponds to each axis of the 3D volume
    shuffled_mask_lesion_indices=np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            shuffled_mask_lesion_indices[j,
                i] = mask_lesion_indices[j, randidx[i]]
    shuffled_mask_lesion_indices= np.asarray(shuffled_mask_lesion_indices, dtype = int)

    # mask out all lesion indices to get all healthy indices
    tmp= copy.deepcopy(invols[0])
    tmp[tmp > 0]= 1
    tmp[tmp <= 0]= 0
    tmp= np.multiply(tmp, 1-mask)

    healthy_brain_indices= np.nonzero(tmp)
    healthy_brain_indices= np.asarray(healthy_brain_indices, dtype = int)
    num_healthy_indices= len(healthy_brain_indices[0])

    randidx0= rng.sample(range(0, num_healthy_indices), num_patches)
    # here, 3 corresponds to each axis of the 3D volume
    shuffled_healthy_brain_indices= np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            shuffled_healthy_brain_indices[j,
                i] = healthy_brain_indices[j, randix0[i]]
    shuffled_healthy_brain_indices= np.asarray(shuffled_healthy_brain_indices, dtype = int)

    newidx = np.concatenate([shuffled_mask_lesion_indices,
                            shuffled_healthy_brain_indices], axis = 1)

    CT_matsize= (2*num_patches, patchsize[0], patchsize[1], num_channels)
    Mask_matsize= (2*num_patches, patchsize[0], patchsize[1], 1)

    CTPatches= np.ndarray(CT_matsize, dtype=np.float16)
    MaskPatches= np.ndarray(Mask_matsize, dtype=np.float16)

    for i in range(0, 2*num_patches):
        I= newidx[0, i]
        J= newidx[1, i]
        K= newidx[2, i]

        for c in range(num_channels):
            CTPatches[i, : , : , c] = invols[c][I - dsize[0]: I + dsize[0] + 1,
                                              J - dsize[1]: J + dsize[1] + 1,
                                              K]

        MaskPatches[i, : , : , 0] = mask[I - dsize[0]: I + dsize[0] + 1,
                                       J - dsize[1]:J + dsize[1] + 1,
                                       K]

    CTPatches = np.asarray(CTPatches, dtype=np.float16)
    MaskPatches = np.asarray(MaskPatches, dtype=np.float16)

    return CTPatches, MaskPatches



def CreatePatchesForTraining(atlasdir, patchsize, max_patch=150000, num_channels=1):
    '''

    Params:
        - TODO
        - healthy: bool, False if not going over the healthy dataset, true otherwise

    '''

    # get filenames
    ct_names = os.listdir(atlasdir)
    mask_names = os.listdir(atlasdir)

    ct_names = [x for x in ct_names if "CT" in x]
    mask_names = [x for x in mask_names if "mask" in x]

    ct_names.sort()
    mask_names.sort()

    numatlas = len(ct_names)

    patchsize = np.asarray(patchsize, dtype=int)
    padsize = np.max(patchsize + 1) / 2

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
    print("Allowed total number of patches =", total_num_patches)

    CT_matsize = (total_num_patches, patchsize[0], patchsize[1], num_channels)
    Mask_matsize = (total_num_patches, patchsize[0], patchsize[1], 1)
    CTPatches = np.zeros(CT_matsize, dtype=np.float16)
    MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

    indices = [x for x in range(total_num_patches)]
    indices = shuffle(indices, random_state=0)
    cur_idx = 0

    for i in tqdm(range(0, numatlas)):
        ctname = ct_names[i]
        ctname = os.path.join(atlasdir, ctname)

        temp = nib.load(ctname)
        ct = temp.get_data()
        ct = np.asarray(ct, dtype=np.float16)

        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nib.load(maskname)
        mask = temp.get_data()
        mask = np.asarray(mask, dtype=np.float16)

        ctt = PadImage(ct, padsize)
        maskt = PadImage(mask, padsize)

        invols = [ctt] # can handle multi-channel input here

        CTPatchesA, MaskPatchesA = get_patches(invols,
                                               maskt,
                                               patchsize,
                                               single_subject_num_patches,
                                               num_channels,)

        CTPatchesA = np.asarray(CTPatchesA, dtype=np.float16)
        MaskPatchesA = np.asarray(MaskPatchesA, dtype=np.float16)


        for ct_patch, mask_patch in zip(CTPatchesA, MaskPatchesA):
            CTPatches[indices[cur_idx], :, :, :] = ct_patch
            MaskPatches[indices[cur_idx], :, :, :] = mask_patch
            cur_idx += 1

    return (CTPatches, MaskPatches)

                                            

"""

# This might be faster than the above implementation, but need to verify.
# For now, disabling this.

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

    fuzzy_edge = 10
    blood_HU_range = range(-fuzzy_edge + 13, fuzzy_edge + 75 + 1)

    healthy_coords, lesion_coords = get_center_coords(invols[0], mask, ratio)

    # print("Ratio: {}\nLength Healthy coords: {}\nMaxpatch: {}".format(ratio, len(healthy_coords), maxpatch))

    if ratio == 1:
        num_patches = np.minimum(maxpatch, len(healthy_coords))
    else:
        num_patches = np.minimum(maxpatch, len(lesion_coords))

    # print("NUM PATCHES:", num_patches)

    # allocate ndarray of maxpatch size
    ct_tensor_shape = (num_patches, patchsize[0], patchsize[1], num_channels)
    mask_tensor_shape = (num_patches, patchsize[0], patchsize[1], 1)
    CTPatches = np.ndarray(ct_tensor_shape, dtype = np.float16)
    MaskPatches = np.zeros(ct_tensor_shape, dtype = np.float16)

    # radius from the center coord to gather patches from
    patch_radius = patchsize // 2

    # set up iterators
    healthy_patch_iter = iter(healthy_coords)
    lesion_patch_iter = iter(lesion_coords)

    # order of indices in which to place patches
    # this pre-shuffles the data
    target_indices = [x for x in range(num_patches)]
    target_indices = shuffle(target_indices, random_state = 0)

    # extract patches
    for counter, i in enumerate(target_indices):
        # swap over to healthy if past ratio limit
        if counter >= (1-ratio) * num_patches:
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

        '''
        # visualize a patch, then exit

        print("\n\nDEBUG HISTOGRAMS \n\n")
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)

        imgplot = plt.imshow(invols[0][x_1:x_2, y_1:y_2, z])
        a.set_title("CT patch")

        # a = fig.add_subplot(1,2,2)
        # maskplot = plt.imshow(mask[x_1:x_2, y_1:y_2, z])
        # a.set_title("Mask patch")

        hist, bin_edges = np.histogram(invols[0][x_1:x_2, y_1:y_2, z])
        a = fig.add_subplot(1,2,2)
        histplot = plt.hist(hist)
        a.set_title("Intensity histogram")

        plt.show()

        cmd = input()
        if cmd == "q":
            sys.exit()


        '''

        # place patches in ndarrays
        for c in range(num_channels):
            CTPatches[i, : , : , c] = invols[c][x_1: x_2, y_1: y_2, z]

        # CTPatches[i,:,:,:][np.invert(np.isin(CTPatches[i,:,:,:], blood_HU_range))] = 0

        MaskPatches[i, : , : , 0] = mask[x_1: x_2, y_1: y_2, z]

    return CTPatches, MaskPatches


def CreatePatchesForTraining(atlasdir, patchsize, unskullstrippeddir = None,
                             max_patch = 150000, num_channels = 1, healthy = False,
                             linear_downscaling = False):
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
        ct_names=os.listdir(atlasdir)
        ct_names=[x for x in ct_names if "CT" in x]
        ct_names.sort()

        numatlas=len(ct_names)

        patchsize=np.asarray(patchsize, dtype = int)
        padsize=np.max(patchsize + 1) / 2

        # hard coded for now
        healthy_ratio=1

        total_num_patches=max_patch

        print("Total number of patches:", max_patch)
        print("Number atlases:", numatlas)

        single_subject_num_patches=total_num_patches // numatlas
        print("Allowed total number of patches =", total_num_patches)
        print("Number of patches per image =", single_subject_num_patches)

        CT_matsize=(total_num_patches,
                      patchsize[0], patchsize[1], num_channels)
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

        if linear_downscaling:
            CTPatches /= np.max(CTPatches)
            CTPatches[np.where(CTPatches < 0)] = 0

        return (CTPatches, MaskPatches)

    else:
        # get filenames
        ct_names = os.listdir(atlasdir)
        mask_names = os.listdir(atlasdir)

        ct_names = [x for x in ct_names if "CT" in x]
        mask_names = [x for x in mask_names if "mask" in x]

        ct_names.sort()
        mask_names.sort()

        numatlas = len(ct_names)

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
        print("Allowed total number of patches =", total_num_patches)

        CT_matsize = (total_num_patches,
                      patchsize[0], patchsize[1], num_channels)
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

            maskname = mask_names[i]
            maskname = os.path.join(atlasdir, maskname)
            temp = nib.load(maskname)
            mask = temp.get_data()
            mask = np.asarray(mask, dtype=np.float16)

            dim = ct.shape

            ctt = PadImage(ct, padsize)
            maskt = PadImage(mask, padsize)

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

        if linear_downscaling:
            CTPatches /= np.max(CTPatches)
            CTPatches[np.where(CTPatches < 0)] = 0

        return (CTPatches, MaskPatches)
"""
