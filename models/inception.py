from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, GlobalAveragePooling2D, add
from keras.layers.merge import Concatenate
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def get_inception_layer(prev_layer, ds=2):
    a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    
    b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    b = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')(b)
    
    c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')(prev_layer)
    c = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(c)

    d = AveragePooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    d = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(d)
    
    e = MaxPooling2D((3,3), strides=(1,1), padding='same')(prev_layer)
    e = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')(e)

    out_layer = concatenate([a,b,c,d,e],axis=-1)

    return out_layer

def inception(num_channels, ds=2,lr=1e-4):

    inputs = Input((None, None, num_channels))

    x = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')(inputs)
    x = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')(x)

    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)
    x = get_inception_layer(x, ds)

    x = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)

    # binary crossentropy with dice as a metric 
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[dice_coef],
                  loss=binary_crossentropy)
                  #loss=weighted_bce)

    return model

def dual_pass_inception(num_channels, lr=1e-4):
    inputs = Input((None, None, num_channels))

    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(x)

    x = get_inception_layer(x)
    x = get_inception_layer(x)
    x = get_inception_layer(x)

    x = Conv2D(32, (5,5), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(16, (3,3), strides=(1,1), activation='relu', padding='same')(x)
    sigmoid_mask_1 = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)

    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(sigmoid_mask_1)
    x = Conv2D(64, (3,3), strides=(1,1), activation='relu', padding='same')(x)

    x = get_inception_layer(x)
    x = get_inception_layer(x)
    x = get_inception_layer(x)

    x = Conv2D(32, (5,5), strides=(1,1), activation='relu', padding='same')(x)
    x = Conv2D(16, (3,3), strides=(1,1), activation='relu', padding='same')(x)
    sigmoid_mask_2 = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(x)

    
    model = Model(inputs=inputs, outputs=sigmoid_mask_2)

    # binary crossentropy with dice as a metric 
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[dice_coef],
                  loss='binary_crossentropy')

    return model

def weighted_bce(y_true, y_pred, smooth=1):
    bce = binary_crossentropy(y_true, y_pred)

    y_true_f = K.cast(K.flatten(y_true), dtype=tf.int32)
    y_pred_f = K.cast(K.round(K.flatten(y_pred)), dtype=tf.int32)

    c_matrix = tf.confusion_matrix(y_true_f, y_pred_f, num_classes=2)

    true_positive = c_matrix[1,1]
    false_positive = c_matrix[1,0]
    true_negative = c_matrix[0,0]
    false_negative = c_matrix[0,1]
    
    sensitivity = (smooth + true_positive) / (smooth + true_positive + false_negative)
    specificity = (smooth + true_negative) / (smooth + true_negative + false_positive)

    sensitivity = K.cast(sensitivity, dtype=tf.float32)
    specificity = K.cast(specificity, dtype=tf.float32)

    # bce plus false positive rate plus false negative rate
    # this weights the bce by the FPR and FNR
    return bce + (1-specificity) + (1-sensitivity)


def tversky_loss(y_true, y_pred, image_dims=2):
    '''
    Ref: salehi17, "Tversky loss function for image segmentation"
    Score is computed for each class separately and then summed

    alpha=beta=0.5: Dice coefficient
    alpha=beta=1: tanimoto coefficient (aka Jaccard)
    alpha+beta=1: produces set of F*-scores
    Implemented by E. Moebel, 06/04/18
    
    ''' 

    if image_dims == 2:
        sum_dims = (0,1,2)
    elif image_dims == 3:
        sum_dims = (0,1,2,3)

    # in the reference, these parameters gave the best dice score
    alpha = 0.5
    beta = 0.5

    ones = K.ones(K.shape(y_true))
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0*g0, sum_dims)
    den = num + alpha*K.sum(p0*g1, sum_dims) + beta * K.sum(p1*g0, sum_dims)

    T = K.sum(num/den)

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.round(K.flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
