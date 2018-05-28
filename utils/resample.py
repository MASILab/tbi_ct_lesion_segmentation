import os

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
