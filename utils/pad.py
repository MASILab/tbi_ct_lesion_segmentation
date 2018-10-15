import numpy as np

def pad_image(img_data):
    '''
    Pads the image to the nearest greater multiple of 16.
    This is due to the downsample/upsample count in the Unet.
    '''

    # pad to nearest greater multiple of 2**NUM_DOWNSAMPLES
    NUM_DOWNSAMPLES = 4
    scaling = 2**NUM_DOWNSAMPLES
    target_dims = [int(np.ceil(x/scaling)) * scaling for x in img_data.shape]

    left_pad = round(float(target_dims[0] - img_data.shape[0]) / 2)
    right_pad = round(float(target_dims[0] - img_data.shape[0]) - left_pad)
    top_pad = round(float(target_dims[1] - img_data.shape[1]) / 2)
    bottom_pad = round(float(target_dims[1] - img_data.shape[1]) - top_pad)
    front_pad = round(float(target_dims[2] - img_data.shape[2]) / 2)
    back_pad = round(float(target_dims[2] - img_data.shape[2]) - front_pad)

    pads = ((left_pad, right_pad),
            (top_pad, bottom_pad),
            (front_pad, back_pad))

    new_img = np.zeros((target_dims))
    new_img[:,:,:] = np.pad(img_data[:,:,:], pads, 'constant', constant_values=0)

    return new_img
