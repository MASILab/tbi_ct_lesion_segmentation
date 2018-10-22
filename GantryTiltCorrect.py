#!/usr/bin/python2.7

import os
import os.path
import sys
import shutil
import dicom
import math
import subprocess
import nibabel as nib
import numpy
from PIL import Image
from scipy import ndimage
from pylab import *


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("wrong input argvs")
        sys.exit()

    arg1, arg2 = sys.argv[1:]

    # make temp dir
    tmp_dir = 'temp'
    tmp_dcm = 'tmp.dcm'
    tmp_nii = 'tmp.nii'
    reg_nii = 'reg.nii'
    tmp_mat = 'tmp.mat'

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    if os.path.isdir(arg1):
        in_dir = arg1
        out_dir = arg2
        if os.path.exists(out_dir):
            shutil.rmtree(arg2)
        # copy the dicoms into the output dir
        shutil.copytree(arg1, arg2)

        # load the dicom names
        filenames = os.listdir(in_dir)
        filenames = [x for x in filenames if '.dcm' in x]
        instanceNumbers = []

        sorted_dicoms = ['NA'] * len(filenames)

        # sort the dicoms
        for filename in filenames:
            # print filename

            dataset = dicom.read_file(os.path.join(out_dir, filename))
            # check the instance number
            # instanceNumbers.append(dataset.data_element("InstanceNumber").value)
            sorted_dicoms[dataset.data_element(
                "InstanceNumber").value-1] = filename

        # get the first slice location
        first_dicom = dicom.read_file(os.path.join(out_dir, sorted_dicoms[0]))
        first = first_dicom.data_element("SliceLocation").value

        # get the pixel spacing
        pixel_spacing = first_dicom.data_element("PixelSpacing").value
        delta_y = pixel_spacing[1]

        # get the gantry tilt
        gantry_tilt = first_dicom.data_element("GantryDetectorTilt").value

        for index in range(len(filenames)):
            # get the slice location for each dicom
            dicom_img = dicom.read_file(
                os.path.join(out_dir, sorted_dicoms[index]))
            location = dicom_img.data_element("SliceLocation").value

            # compute the offset
            offset = math.tan(abs(gantry_tilt)*math.pi/180) * \
                (location-first)/delta_y

            #print(index)
            H = array([[1, 0, offset], [0, 1, 0], [0, 0, 1]])

            print H
            continue

            #print(dicom_img.pixel_array[256, 256])
            reg = ndimage.affine_transform(
                dicom_img.pixel_array, H[:2, :2], (H[0, 2], H[1, 2]), cval=-1024)
            # print reg.shape
            #print(reg[256, 256])
            # for n,val in enumerate(dicom_img.pixel_array.flat):
            #	dicom_img.pixel_array.flat[n]=reg.flat[n]

            numpy.copyto(dicom_img.pixel_array, reg)
            #dicom_img.pixel_array = reg.copy()
            dicom_img.PixelData = dicom_img.pixel_array.tostring()
            #print(dicom_img.pixel_array[256, 256])
            dicom_img.save_as(os.path.join(out_dir, sorted_dicoms[index]))

    #		# save the transofrmation matrix
    #		mtx = open(tmp_mat,'w')
    #		mtx.write('1 0 0 0\n')
    #		mtx.write('0 1 0 ' + repr(offset) + '\n')
    #		mtx.write('0 0 1 0\n')
    #		mtx.write('0 0 0 1')
    #		mtx.close()
    #
    #		# conver dicom to nifti
    #		src = os.path.join(out_dir,sorted_dicoms[ind])
    #		tar = os.path.join(tmp_dir,tmp_dcm)
    #		shutil.copyfile(src,tar)
    #
    #		proc = subprocess.Popen('dcm2nii -c N -d N -e N -f Y -g N -i N -p N %s' % repr(tar), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    #		nii = os.path.join(tmp_dir,tmp_nii)
    #		reg = os.path.join(tmp_dir,reg_nii)
    #		#proc = subprocess.Popen('flirt -2D -in temp/tmp.nii -ref temp/tmp.nii -applyxfm -init tmp.mat -out /temp/reg.nii', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    #		#program  = "flirt"
    #		#arguments = ["-2D", "-in", "temp/tmp.nii", "-ref", "temp/tmp.nii", "-applyxfm", "-init", "tmp.mat", "-out", "temp/reg.nii"]
    #		#command = [program]
    #		#command.extend(arguments)
    #		#proc = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    #		#print ind
    #		#command = "flirt -2D -in /Users/user/Projects/GantryTiltCorrect/temp/tmp.nii -ref /Users/user/Projects/GantryTiltCorrect/temp/tmp.nii -applyxfm -init /Users/user/Projects/GantryTiltCorrect/tmp.mat -out /Users/user/Projects/GantryTiltCorrect/temp/reg.nii"
    #		#print command
    #		#os.system(command)
    #		#os.system("flirt -2D -in /Users/user/Projects/GantryTiltCorrect/temp/tmp.nii -ref /Users/user/Projects/GantryTiltCorrect/temp/tmp.nii -applyxfm -init /Users/user/Projects/GantryTiltCorrect/tmp.mat -out /Users/user/Projects/GantryTiltCorrect/temp/reg.nii")
    #
    #		flirt_call = 'flirt -2D -in ' +nii + ' -ref ' +  nii + ' -applyxfm -init ' + tmp_mat + ' -out ' + reg
    #		#print flirt_call
    #		#os.system(flirt_call)
    #		porc = subprocess.Popen(flirt_call,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #
    #		#os.remove(src)
    #		#os.remove(tar)
    #		#os.remove(nii)
    #		#os.remove(reg)
    #		cwd = os.getcwd()
    #		os.chdir(os.path.join(cwd,'temp'))
    #
    #	#	I01=os.path.join(cwd,'temp/reg.nii.gz')
    # print I01
    #		# load the register nii
    #		img = nib.load('reg.nii.gz')
    #		data = img.get_data()
    #		data.shape
