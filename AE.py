import tensorflow as tf
from tensorflow.keras import Model
from tslearn.metrics import dtw
from tensorflow.keras.losses import MeanSquaredError

def L2Loss(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)

class Autoencoder(Model):
    def __init__(self, encoder, decoder, num_steps):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_steps = num_steps

    def call(self, inputs):
        encoded = self.encoder(inputs)
        reconstruction = self.decoder(encoded)
        return reconstruction

    @tf.function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            encoded = self.encoder(data)
            reconstruction = self.decoder(encoded)
            reconstruction_loss = L2Loss(data, reconstruction)
            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss
        }

def errors(modello, data):
    predictions = modello.predict(data)
    
    data = tf.cast(data, tf.float64)
    predictions = tf.cast(predictions, tf.float64)

    if len(data.shape) == 4:
        data = tf.reshape(data, (data.shape[0], data.shape[1]*data.shape[2], data.shape[3]))
        predictions = tf.reshape(predictions, (predictions.shape[0], predictions.shape[1]*predictions.shape[2], predictions.shape[3]))

    errori_dtw = []
    errori_mse = []
    errori_l2 = []
    mse = MeanSquaredError()

    for i in range(data.shape[0]):
        score_dtw = dtw(data[i], predictions[i])
        errori_dtw.append(score_dtw)

        score_mse = mse(data[i], predictions[i])
        errori_mse.append(score_mse)

        score_l2 = tf.nn.l2_loss(data[i] - predictions[i])
        errori_l2.append(score_l2)

    return errori_dtw, errori_mse, errori_l2