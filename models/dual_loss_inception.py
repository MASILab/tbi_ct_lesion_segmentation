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

    # instantiate layers to share inputs
    conv1 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv2 = Conv2D(64//ds, (3,3), strides=(1,1), activation='relu', padding='same')

    inception1_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception1_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception1_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception1_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception1_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception1_out_layer = concatenate([a,b,c,d,e],axis=-1)

    inception2_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception2_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception2_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception2_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception2_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception2_out_layer = concatenate([a,b,c,d,e],axis=-1)
    
    inception3_a = Conv2D(64//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_b = Conv2D(96//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_b_2 = Conv2D(128//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    inception3_c = Conv2D(16//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_c_2 = Conv2D(32//ds, (5,5), strides=(1,1), activation='relu', padding='same')
    inception3_d = AveragePooling2D((3,3), strides=(1,1), padding='same')
    inception3_d_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_e = MaxPooling2D((3,3), strides=(1,1), padding='same')
    inception3_e_2 = Conv2D(32//ds, (1,1), strides=(1,1), activation='relu', padding='same')
    inception3_out_layer = concatenate([a,b,c,d,e],axis=-1)

    conv3 = Conv2D(32//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv4 = Conv2D(16//ds, (3,3), strides=(1,1), activation='relu', padding='same')
    conv5 = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')

    # connect inputs and outputs
    unhealthy_input = Input((None, None, num_channels), name='unhealthy_input')

    x = conv1(unhealthy_input) 
    x = conv2(x)

    a = inception1_a(x)
    b = inception1_b(a)
    b = inception1_b_2(b)
    c = inception1_c(b)
    c = inception1_c_2(c)
    d = inception1_d(c)
    d = inception1_d_2(d)
    e = inception1_e(d)
    e = inception1_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    a = inception2_a(x)
    b = inception2_b(a)
    b = inception2_b_2(b)
    c = inception2_c(b)
    c = inception2_c_2(c)
    d = inception2_d(c)
    d = inception2_d_2(d)
    e = inception2_e(d)
    e = inception2_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    a = inception3_a(x)
    b = inception3_b(a)
    b = inception3_b_2(b)
    c = inception3_c(b)
    c = inception3_c_2(c)
    d = inception3_d(c)
    d = inception3_d_2(d)
    e = inception3_e(d)
    e = inception3_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    x = conv3(x)
    x = conv4(x)
    x = conv5(x)
    unhealthy_ouput = Activation('linear', name='unhealthy_output')(x)

    healthy_input = Input((None, None, num_channels, name='healthy_input'))
    x = conv1(healthy_input) 
    x = conv2(x)

    a = inception1_a(x)
    b = inception1_b(a)
    b = inception1_b_2(b)
    c = inception1_c(b)
    c = inception1_c_2(c)
    d = inception1_d(c)
    d = inception1_d_2(d)
    e = inception1_e(d)
    e = inception1_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    a = inception2_a(x)
    b = inception2_b(a)
    b = inception2_b_2(b)
    c = inception2_c(b)
    c = inception2_c_2(c)
    d = inception2_d(c)
    d = inception2_d_2(d)
    e = inception2_e(d)
    e = inception2_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    a = inception3_a(x)
    b = inception3_b(a)
    b = inception3_b_2(b)
    c = inception3_c(b)
    c = inception3_c_2(c)
    d = inception3_d(c)
    d = inception3_d_2(d)
    e = inception3_e(d)
    e = inception3_e_2(e)
    x = concatenate([a,b,c,d,e],axis=-1)

    x = conv3(x)
    x = conv4(x)
    x = conv5(x)
    healthy_ouput = Activation('linear', name='healthy_output')(x)
    
    model = Model(inputs=[unhealthy_inputs, healthy_inputs], 
                  outputs=[unhealthy_outputs, healthy_outputs])

    # binary crossentropy with dice as a metric 
    model.compile(optimizer=Adam(lr=lr),
                  metrics=[dice_coef],
                  loss={'unhealthy_output':true_positive_rate, 
                        'healthy_output':false_positive_rate}, 
                  loss_weights={'unhealthy_output':1.,
                                'healthy_output':1.,})

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

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.round(K.flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
