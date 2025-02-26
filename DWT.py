from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Conv1D, Conv1DTranspose, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from AE import Autoencoder, L2Loss


def create_dwt2v_img_encoder(img_cols, img_rows, time_steps, num_channels, encoder_params, decoder_params, l2_reg=0.005):
    model = tf.keras.Sequential(name='Encoder')
    model.add(tf.keras.layers.Input(shape=(img_cols, img_rows, num_channels)))
    
    model.add(Conv2D(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    if decoder_params['num_layers'] > 1:
        for i in range(1, decoder_params['num_layers']):
            model.add(Conv2D(decoder_params[f"conv_layer_{decoder_params['num_layers']-i}"], decoder_params[f"stride_layer_{decoder_params['num_layers']-i}"], padding='same', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(time_steps * encoder_params['embedding_layer'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))

    return model

def create_dwt2v_img_decoder(img_cols, img_rows, time_steps, num_channels, encoder_params, decoder_params, l2_reg=0.005):
    dim=encoder_params['embedding_layer']
    model = Sequential(name='Decoder')
    model.add(tf.keras.layers.Input(shape=(time_steps*encoder_params['embedding_layer'],)))
    
    if decoder_params['num_layers']>1:
        model.add(tf.keras.layers.Dense(img_cols*img_rows*decoder_params['conv_layer_1'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(tf.keras.layers.Reshape((img_cols, img_rows, decoder_params['conv_layer_1'])))
    else:
        model.add(tf.keras.layers.Dense(img_cols*img_rows*num_channels, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(tf.keras.layers.Reshape((img_cols, img_rows, num_channels)))

    if(decoder_params['num_layers']>1):
        for i in range(1, decoder_params['num_layers']):
            model.add(Conv2DTranspose(decoder_params[f"conv_layer_{decoder_params['num_layers']-i}"], decoder_params[f"stride_layer_{decoder_params['num_layers']-i}"], padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(Conv2DTranspose(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    else: 
        model.add(Conv2DTranspose(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    return model

def create_dwt2v_vec_encoder(vec_len, time_steps, num_channels, encoder_params, decoder_params, l2_reg=0.005):
    model = tf.keras.Sequential(name='Encoder')
    model.add(tf.keras.layers.Input(shape=(vec_len, num_channels)))
    
    model.add(Conv1D(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    if decoder_params['num_layers']>1:
        for i in range(1, decoder_params['num_layers']):
            model.add(Conv1D(decoder_params[f"conv_layer_{decoder_params['num_layers']-i}"], decoder_params[f"stride_layer_{decoder_params['num_layers']-i}"], padding='same', kernel_regularizer=l2(l2_reg)))
            model.add(BatchNormalization())
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(time_steps*encoder_params['embedding_layer'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))

    return model

def create_dwt2v_vec_decoder(vec_len, time_steps, num_channels, encoder_params, decoder_params, l2_reg=0.005):
    dim=encoder_params['embedding_layer']
    model = Sequential(name='Decoder')
    model.add(tf.keras.layers.Input(shape=(time_steps*encoder_params['embedding_layer'],)))
    if decoder_params['num_layers']>1:
        model.add(tf.keras.layers.Dense(vec_len*decoder_params['conv_layer_1'], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(tf.keras.layers.Reshape((vec_len, decoder_params['conv_layer_1'])))
    else:
        model.add(tf.keras.layers.Dense(vec_len*num_channels, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(tf.keras.layers.Reshape((vec_len, num_channels)))

    if(decoder_params['num_layers']>1):
        for i in range(1, decoder_params['num_layers']):
            model.add(Conv1DTranspose(decoder_params[f'conv_layer_{i}'], decoder_params[f'stride_layer_{i}'], padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(Conv1DTranspose(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    else: 
        model.add(Conv1DTranspose(num_channels, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    return model

def create_dw2v_vec_autoencoder(vec_len, time_steps, num_channels, encoder_params, decoder_params):
    encoder_ae = create_dwt2v_vec_encoder(vec_len, time_steps, num_channels, encoder_params, decoder_params)
    decoder_ae = create_dwt2v_vec_decoder(vec_len, time_steps, num_channels, encoder_params, decoder_params)
    ae = Autoencoder(encoder_ae, decoder_ae, num_steps=vec_len)
    ae.build((None, vec_len, num_channels))
    ae.compile(optimizer=tf.keras.optimizers.Adam(), loss=L2Loss)
    return ae

def create_dw2v_img_autoencoder(img_cols, img_rows, time_steps, num_channels, encoder_params, decoder_params):
    encoder_ae = create_dwt2v_img_encoder(img_cols, img_rows, time_steps, num_channels, encoder_params, decoder_params)
    decoder_ae = create_dwt2v_img_decoder(img_cols, img_rows, time_steps, num_channels, encoder_params, decoder_params)
    ae = Autoencoder(encoder_ae, decoder_ae, num_steps=img_cols)
    ae.build((None, img_cols, img_rows, num_channels))
    ae.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=5.0), loss=L2Loss)
    return ae