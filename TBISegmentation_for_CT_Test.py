import nibabel as nifti
import glob
import os
import sys
import random
import math
import copy
import numpy as np
from scipy import ndimage
from scipy.signal import argrelextrema
import argparse
#import tensorflow as tf
from keras.models import load_model
from keras import backend as K
import statsmodels.api as sm
import time
from scipy import ndimage
from tqdm import tqdm
from distutils import spawn
from termcolor import cprint
from models.inception import weighted_bce, dice_coef, dice_coef_loss, tversky_loss
import tempfile
import h5py
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def TestExecutable(name):
    x = spawn.find_executable(name)
    if x is None:
        y = 'ERROR: ' + name + ' not found. Please install AFNI, FSL and add them to PATH.'
        cprint(y, 'red')
        sys.exit()
    else:
        y = 'I found ' + name + ' at ' + x
        cprint(y, 'green')


def ApplyModel2D_multichannel(vol1, model, num_channels):
    dim = vol1.shape[:-1]
    dim2D = (1, dim[0], dim[1], num_channels)
    dim2D = np.asarray(dim2D, dtype=int)
    invol2D = np.zeros(dim2D, dtype=float)
    outvol = np.zeros(dim, dtype=float)

    for k in tqdm(range(0, dim[2])):
        for c in range(num_channels):
            invol2D[0, :, :, c] = vol1[:, :, k, c]
        pred = model.predict(invol2D)
        outvol[:, :, k] = pred[0, :, :, 0]

    return outvol


def ApplyModel2D(vol1, model):

    dim = vol1.shape
    dim2D = (1, dim[0], dim[1], 1)
    dim2D = np.asarray(dim2D, dtype=int)
    invol2D = np.zeros(dim2D, dtype=float)
    outvol = np.zeros(dim, dtype=float)

    for k in tqdm(range(0, dim[2])):
        invol2D[0, :, :, 0] = vol1[:, :, k]
        pred = model.predict(invol2D)
        outvol[:, :, k] = pred[0, :, :, 0]

    return outvol


def CheckCT(ct):
    temp = copy.deepcopy(ct)
    temp = temp[np.nonzero(temp)]
    temp = np.asarray(temp, dtype=float)
    q1 = np.percentile(temp, 99.0)
    q2 = np.percentile(temp, 5.0)
    print(
        "Subject CT has 5th and 99th percentile range [%.2f,%.2f]" % (q2, q1))
    if q1 > 3000:
        cprint('ERROR: Subject image does not look like a proper CT image.', 'red')
        q3 = np.amin(temp)
        q4 = np.amax(temp)
        x = 'ERROR: Intensities should be in the range [-1024,2048]. I found min and max as [' \
            + str(q3) + ',' + str(q4) + ']'
        cprint(x, 'red')
        cprint('ERROR: Please check the CT image and make sure it is within acceptable CT HU range.', 'red')
        sys.exit()


def Suffix(x):
    if x == 1:
        s = 'st'
    elif x == 2:
        s = 'nd'
    elif x == 3:
        s = 'rd'
    else:
        s = 'th'
    return s


def remove_ext(inputname):
    l = len(inputname)
    try:

        if inputname[-4:] == ".nii":
            outname = inputname[0:l-4]
            return outname
    except:
        return inputname
    try:
        if inputname[-7:] == ".nii.gz":
            outname = inputname[0:l - 7]
            return outname
    except:
        return inputname


parser = argparse.ArgumentParser(
    description='Prediction of TBI segmentation for CT')

parser.add_argument('--models', action='append', required=True, type=list, nargs='+', dest='MODELS',
                    help='Learnt models (.h5) files')
parser.add_argument('--ct', action='store', required=True,
                    dest='T1', help='T1 Image (skullstripped)')
parser.add_argument('--o', action='store', dest='OUTDIR', required=True,
                    help='Output directory where the resultant membership is written')

if len(sys.argv) < 2:
    parser.print_usage()
    sys.exit(1)

results = parser.parse_args()
im1 = os.path.expanduser(results.T1)
im1 = os.path.realpath(im1)

results.OUTDIR = os.path.expanduser(results.OUTDIR)
results.OUTDIR = os.path.realpath(results.OUTDIR)
base = os.path.basename(im1)
base = remove_ext(base)
outname1 = base + "_CNNLesionMembership.nii.gz"
outname1 = os.path.join(results.OUTDIR, outname1)
#outname2=base + "_CNNStripMask.nii.gz"
# outname2=os.path.join(results.OUTDIR,outname2)


x = results.MODELS
x = x[0]
models = []
for i in range(0, len(x)):
    models.append(''.join(x[i]))

print("%d models found at" % (len(models)))
for i in range(0, len(models)):
    models[i] = os.path.expanduser(models[i])
    models[i] = os.path.abspath(models[i])
    print(models[i])


# These 3 are must to run this sript

TestExecutable('3dresample')
TestExecutable('fslmaths')


tmpdir = tempfile.mkdtemp()
tmpdir = tmpdir + os.path.sep
print("Temporary directory :" + tmpdir)

'''
# FOR NOW: SKIP REORIENTATION!!!
x=tmpdir + 'ct.nii'
cmd = '3dresample -orient RAI -inset ' + im1 + ' -prefix ' + x
os.system(cmd)
im1=x
'''


print("CT image         = " + im1)
print("Output directory = " + results.OUTDIR)


temp1 = nifti.load(im1)
vol1 = temp1.get_data()
im_size = vol1.shape
num_channels = im_size[-1]
num_dims_im_size = len(im_size)
if num_dims_im_size == 4:
    im_size = im_size[:-1]

CheckCT(vol1)


x = np.array(im_size)
y = np.array(len(models))
y = np.asarray(y, dtype='int')
im_size2 = np.append(x, y)
outvol = np.zeros(im_size2)

newmodels = []
for i in range(0, len(models)):
    src = models[i]
    name = str(i+1) + '.h5'
    dst = os.path.join(tmpdir, name)
    shutil.copy(src, dst)
    newmodels.append(dst)


for t in range(0, len(models)):
    start = time.time()
    '''
    model = load_model(newmodels[t])
    '''
    # for dice
    model = load_model(newmodels[t], custom_objects={'weighted_bce': weighted_bce,
                                                     'dice_coef': dice_coef})

    with h5py.File(newmodels[t], 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']
    print("***Applying Model***")
    if num_dims_im_size == 3:
        mem = ApplyModel2D(vol1, model)
    else:
        mem = ApplyModel2D_multichannel(vol1, model, num_channels=num_channels)

    outvol[:, :, :, t] = mem

    elapsed = time.time() - start

    print("Time taken for %d%s atlas= %.2f seconds" %
          (t+1, Suffix(t+1), elapsed))

outvol = np.average(outvol, axis=3)
#outvol = outvol/100.0

# save the whole membership
temp3 = nifti.Nifti1Image(outvol, temp1.affine, temp1.header)
print("Writing " + outname1)
nifti.save(temp3, outname1)


shutil.rmtree(tmpdir)

K.clear_session()
