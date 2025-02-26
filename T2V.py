from tensorflow.keras.layers import Layer, Input, Conv1DTranspose, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf
from AE import Autoencoder

class Time2Vec(Layer):
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(Time2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.output_dim - 1),
                                 initializer='uniform',
                                 trainable=True,
                                 dtype=K.floatx())

        self.P = self.add_weight(name='P',
                                 shape=(input_shape[1], self.output_dim - 1),
                                 initializer='uniform',
                                 trainable=True,
                                 dtype=K.floatx())

        self.w = self.add_weight(name='w',
                                 shape=(input_shape[- 1], 1),
                                 initializer='uniform',
                                 trainable=True,
                                 dtype=K.floatx())

        self.p = self.add_weight(name='p',
                                 shape=(input_shape[1], 1),
                                 initializer='uniform',
                                 trainable=True,
                                 dtype=K.floatx())

        super(Time2Vec, self).build(input_shape)

    def call(self, x):
        original = tf.matmul(x, self.w) + self.p
        sin_trans = tf.sin(tf.matmul(x, self.W) + self.P)
        return tf.concat([sin_trans, original], axis=-1)

def create_t2v_encoder(time_steps, num_features, encoder_params, l2_reg=0.005):
    model = Sequential(name='Encoder')
    model.add(Input(shape = (time_steps, num_features)))
    model.add(Time2Vec(output_dim=encoder_params['embedding_layer']))
    model.add(Flatten())
    return model

def create_t2v_decoder(time_steps, num_features, decoder_params, encoder_params, l2_reg=0.005):
    dim=encoder_params['embedding_layer']
    model = Sequential(name='Decoder')
    model.add(Reshape((time_steps, dim), input_shape=(time_steps*dim,)))
    
    if(decoder_params['num_layers']>1):
        for i in range(1, decoder_params['num_layers']):
            model.add(Conv1DTranspose(decoder_params[f'conv_layer_{i}'], decoder_params[f'stride_layer_{i}'], padding='same', kernel_regularizer=l2(l2_reg)))
        model.add(Conv1DTranspose(num_features, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    else: 
        model.add(Conv1DTranspose(num_features, decoder_params['last_kernel'], padding='same', kernel_regularizer=l2(l2_reg)))
    return model

def create_t2v_autoencoder(time_step, num_features, encoder_params, decoder_params):
    encoder_ae = create_t2v_encoder(time_step, num_features, encoder_params)
    decoder_ae = create_t2v_decoder(time_step, num_features, decoder_params, encoder_params)
    ae = Autoencoder(encoder_ae, decoder_ae, num_steps=time_step)
    ae.build((None, time_step, num_features))
    ae.compile(optimizer='adam')
    return ae