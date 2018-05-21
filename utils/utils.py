from time import strftime
import os
import argparse

def preprocess(filename, src_dir, preprocess_root_dir):
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
        skullstrip(filename, src_dir, SKULLSTRIP_DIR)
        n4biascorrect(filename, SKULLSTRIP_DIR, N4_DIR)
        resample(filename, N4_DIR, RESAMPLE_DIR)
        orient(filename, RESAMPLE_DIR, RAI_DIR)
    elif "mask" in filename or "multiatlas" in filename:
        resample(filename, src_dir, RESAMPLE_DIR)
        orient(filename, RESAMPLE_DIR, RAI_DIR)

    return RAI_DIR


def GetPatches(invols, mask, patchsize, maxpatch, num_channels, loss):
    # now supports multi-channel patches

    rng = random.SystemRandom()
    mask = np.asarray(mask, dtype=np.float32)
    patchsize = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patchsize / 2).astype(dtype=int)

    indx = np.nonzero(mask)
    indx = np.asarray(indx, dtype=int)
    L = len(indx[0])

    L1 = np.minimum(maxpatch, L)
    print("Number of patches used  = %d out of %d (maximum %d)" %(L1,L,maxpatch))
    randindx = rng.sample(range(0, L), L1)
    newindx = np.ndarray((3, L1))
    for i in range(0, L1):
        for j in range(0, 3):
            newindx[j, i] = indx[j, randindx[i]]
    newindx = np.asarray(newindx, dtype=int)

    temp=copy.deepcopy(invols[0])
    temp[temp>0] = 1
    temp[temp<=0] = 0
    temp=np.multiply(temp,1-mask)

    indx0 = np.nonzero(temp)
    indx0 = np.asarray(indx0, dtype=int)
    L = len(indx0[0])
    randindx0 = rng.sample(range(0, L), L1)
    newindx0 = np.ndarray((3, L1))
    for i in range(0, L1):
        for j in range(0, 3):
            newindx0[j, i] = indx0[j, randindx0[i]]
    newindx0 = np.asarray(newindx0, dtype=int)

    newindx = np.concatenate([newindx,newindx0],axis=1)
    #print newindx.shape


    CT_matsize1 = (2*L1, patchsize[0], patchsize[1], num_channels)
    Mask_matsize1 = (2*L1, patchsize[0], patchsize[1], 1)

    if loss == 'mse' or loss == 'mae':
        blurmask = np.zeros(mask.shape, dtype=np.float32)
        for t in range(0, mask.shape[2]):
            if np.ndarray.sum(mask[:, :, t]) > 0:
                blurmask[:, :, t] = ndimage.filters.gaussian_filter(mask[:, :, t], sigma=(1, 1))

        blurmask = np.ndarray.astype(blurmask, dtype=np.float32)
        blurmask[blurmask < 0.0001] = 0
        blurmask = blurmask * 100  # Just to have reasonable looking error values during training, no other reason.
    else:
        blurmask = mask

    CTPatches = np.ndarray(CT_matsize1, dtype=np.float16)
    MaskPatches = np.ndarray(Mask_matsize1, dtype=np.float16)

    for i in range(0, 2*L1):
        I = newindx[0, i]
        J = newindx[1, i]
        K = newindx[2, i]

        for c in range(num_channels):
            CTPatches[i, :, :, c] = invols[c][I - dsize[0]:I +
                                           dsize[0] + 1, J - dsize[1]:J + dsize[1] + 1, K]
        MaskPatches[i, :, :, 0] = blurmask[I - dsize[0]:I +
                                           dsize[0] + 1, J - dsize[1]:J + dsize[1] + 1, K]

    CTPatches = np.asarray(CTPatches, dtype=np.float16)
    MaskPatches = np.asarray(MaskPatches, dtype=np.float16)
    return CTPatches, MaskPatches


def CreatePatchesForTraining(atlasdir, patchsize, unskullstrippeddir=None,\
        max_patch=150000, num_channels=1):
    '''
    This code is identical to the main() function immediately below,
    except instead of training the model it returns a tuple containing
    the CT patches and the mask patches.

    As such, there is no need for the outdir parameter.

    '''

    MaxPatch = max_patch  # Maximum number of patches from a single subject

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
    unskullstripped_names.sort()

    numatlas = len(multiatlas_names)

    patchsize = np.asarray(patchsize, dtype=int)
    padsize = np.max(patchsize + 1) / 2
    f = 0
    for i in range(0, numatlas):
        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        temp = nifti.load(maskname)
        mask = temp.get_data()
        f = f + np.sum(mask)

    f = np.asarray(f, dtype=int)
    print("Total number of lesion patches = " + str(f))
    f = np.minimum(MaxPatch * numatlas, int(str(f)))
    print("Allowed total number of lesion patches = " + str(f))
    CT_matsize = (f, patchsize[0], patchsize[1], num_channels)
    Mask_matsize = (f, patchsize[0], patchsize[1], 1)
    CTPatches = np.zeros(CT_matsize, dtype=np.float16)
    MaskPatches = np.zeros(Mask_matsize, dtype=np.float16)

    ID = time.strftime("%d-%m-%Y") + "_" + time.strftime("%H-%M-%S")
    #print("Unique ID is %s " % (ID))

    x = str(int(patchsize[0])) + "x" + str(int(patchsize[1]))

    count2 = 0
    count1 = 0

    indices = [x for x in range(f)]
    indices = shuffle(indices, random_state=0)
    cur_idx = 0
    
    for i in tqdm(range(0, numatlas)):
        ctname = ct_names[i]
        ctname = os.path.join(atlasdir, ctname)
        #print("Reading %s" % (ctname))
        temp = nifti.load(ctname)
        ct = temp.get_data()
        ct = np.asarray(ct, dtype=np.float16)

        multiatlasname = multiatlas_names[i]
        multiatlasname = os.path.join(atlasdir, multiatlasname)
        #print("Reading %s" % (multiatlasname))
        temp = nifti.load(multiatlasname)
        multiatlas = temp.get_data()
        multiatlas = np.asarray(multiatlas, dtype=np.float16)

        unskullstrippedname = unskullstripped_names[i]
        unskullstrippedname = os.path.join(atlasdir, unskullstrippedname)
        #print("Reading %s" % (unskullstrippedname))
        temp = nifti.load(unskullstrippedname)
        unskullstripped = temp.get_data()
        unskullstripped = np.asarray(unskullstripped, dtype=np.float16)

        maskname = mask_names[i]
        maskname = os.path.join(atlasdir, maskname)
        #print("Reading %s" % (maskname))
        temp = nifti.load(maskname)
        mask = temp.get_data()
        mask = np.asarray(mask, dtype=np.float16)

        dim = ct.shape
        #print("Image size = %d x %d x %d " % (dim[0], dim[1], dim[2]))

        ctt = PadImage(ct, padsize)
        multiatlast = PadImage(multiatlas, padsize)
        unskullstrippedt = PadImage(unskullstripped, padsize)
        maskt = PadImage(mask, padsize)

        CTPatchesA, MaskPatchesA = GetPatches(
            [ctt, multiatlast, unskullstrippedt], maskt, patchsize, MaxPatch, num_channels,opt['loss'])

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
        parser.add_argument('--model', required=True, action='store', dest='model',
                            help='Model Architecture (.json) file')
        parser.add_argument('--weights', required=True, action='store', dest='weights',
                            help='Learnt weights (.hdf5) file')
        parser.add_argument('--result_dst', required=True, action='store', dest='OUT_DIR',
                            help='Output directory where the results are written')
        parser.add_argument('--result_file', required=True, action='store', dest='OUTFILE',
                            help='Output directory where the results are written')
        parser.add_argument('--numcores', required=True, action='store', dest='numcores',
                            default='1', type=int,
                            help='Number of cores to preprocess in parallel with')
    else:
        print("Invalid session. Must be one of \"train\", \"validate\", or \"test\"")
        sys.exit()

    parser.add_argument('--num_channels', required=True, action='store', dest='num_channels',
                        help='Number of channels to include. First is CT, second is atlas,\
                                third is unskullstripped CT')

    return parser.parse_args()


def skullstrip(filename, src_dir, dst_dir, script_path, verbose=0):
    '''
    Skullstrips a CT nifti image into data_dir/preprocessing/skullstripped/

    Params:
        - filename: string, name of file to skullstrip
        - src_dir: string, path to directory where the CT to be skullstripped is
        - dst_dir: string, path to directory where the skullstripped CT is saved
        - script_path: string, path to the file of the skullstrip script
        - verbose: int, 0 for silent, 1 for verbose
    '''

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already skullstripped", filename)
        return

    if verbose == 1:
        print("Skullstripping", filename, "into" + " " + dst_dir)

    call = "sh" + " " + script_path + " " + infile + " " + outfile
    os.system(call)

    if verbose == 1:
        print("Skullstripping complete")


def resample(filename, src_dir, dst_dir, verbose=0):
    '''
    Resamples the CT to 0.5mm x 0.5mm x 5mm with cubic interpolation

    Requires AFNI 3dresample

    Params:
        - filename: string, name of original CT image
        - src_dir: string, path to skullstripped dir
        - dst_dir: string, path to RAI oriented dir
        - verbose: int, 0 for silent, 1 for verbose
    '''
    dims = (0.5, 0.5, 5)  # hard-coded target value

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already resampled", filename)
        return

    if verbose == 1:
        print("Resampling to ", dims, "...")

    call = "3dresample -dxyz" + " " +\
        str(dims[0]) + " " +\
        str(dims[1]) + " " +\
        str(dims[2]) + " " +\
        "-rmode" + " " + "Cu" + " " +\
        "-inset" + " " + infile + " " +\
        "-prefix" + " " + outfile
    os.system(call)

    if verbose == 1:
        print("Resampling complete")


def orient(filename, src_dir, dst_dir, verbose=0):
    '''
    Orients image to RAI using 3dresample into data_dir/preprocessing/rai.

    Requires AFNI 3dresample

    Params:
        - filename: string, name of original CT image
        - src_dir: string, path to skullstripped dir
        - dst_dir: string, path to RAI oriented dir
        - verbose: int, 0 for silent, 1 for verbose
    '''
    target_orientation = "RAI"

    infile = os.path.join(src_dir, filename)
    outfile = os.path.join(dst_dir, filename)

    if os.path.exists(outfile):
        if verbose == 1:
            print("Already oriented", filename)
        return

    if verbose == 1:
        print("Orienting to " + target_orientation+"...")

    call = "3dresample -orient" + " " + target_orientation + " " +\
        "-inset" + " " + infile + " " +\
        "-prefix" + " " + outfile
    os.system(call)

    if verbose == 1:
        print("Orientation complete")


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
