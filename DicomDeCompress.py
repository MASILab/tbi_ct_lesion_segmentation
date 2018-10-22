# DicomDeCompress.py
# Decompress the compressed dicoms

usage = """
Usage:
python DicomDeCompress <DIR>
"""

import sys, os, fnmatch
import subprocess

def locate(in_dir):
    for path, dirs, files in os.walk(os.path.abspath(in_dir)):

        for filename in files:
                yield os.path.join(path,filename)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
	print(usage)
	sys.exit()
        
    arg1 = sys.argv[1]
    
    for file in locate(arg1):
        cmd = '/home/phamdl/Software/bin/gdcmconv -w -i ' + file + ' -o ' +file
	print cmd
	#os.system(cmd)
	proc = subprocess.Popen(cmd,shell=True)
	return_code = proc.wait()
