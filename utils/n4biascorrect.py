import os

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
