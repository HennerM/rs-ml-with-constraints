from models.BaseModel import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import Model, backend as K
import numpy as np


class ConstraintAutoRec(BaseModel):

    def __init__(self, dimensions, **kwargs):
        super().__init__(dimensions, **kwargs)
        self.latent_dims = kwargs.get('latent_dim', 128)
        self.input_dim = dimensions
        self.accuracy_weight = kwargs.get('accuracy_weight', 1.0)
        self.novelty_weight = kwargs.get('novelty_weight', 0.1)
        self.diversity_weight = kwargs.get('diversity_weight', 0.1)
        self.epochs = kwargs.get('epochs', 20)
        self.batch_size = kwargs.get('batch_size', 32)

        self.prepare_model()

    def prepare_model(self):
        rating = Input(shape=(self.input_dim,))
        rating_noisy = Input(shape=(self.input_dim,))

        mask = Input(shape=(self.input_dim,))
        x = rating_noisy
        x = Dense(64, activation='relu')(x)
        latent = Dense(self.latent_dims, name='latent_vector', activation='relu')(x)

        self.encoder = Model(rating_noisy, latent, name='encoder')
        # encoder.summary()

        latent_inputs = Input(shape=(self.latent_dims,), name='decoder_input')
        x = latent_inputs
        x = Dense(64, activation='relu')(x)

        outputs = Dense(self.input_dim, name='decoder_output', activation='sigmoid')(x)
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()

        def constraint_loss(y_true, y_pred):
            return self.augmented_loss(y_true, y_pred, mask, rating_noisy)

        self.model = Model(inputs=[rating, mask, rating_noisy], outputs=self.decoder(self.encoder(rating_noisy)),
                           name='ConstraintAutoRec')
        self.model.compile(optimizer='adam', loss=constraint_loss, metrics=['accuracy'])


    def augmented_loss(self, y_true, y_pred, x_mask, x_noisy):
        # error_constraint = alpha * tf.reduce_sum(estimated * actual)
        novelty_constraint = self.novelty_weight * tf.reduce_sum(y_pred * (tf.reduce_sum(y_pred, 0) / tf.cast(tf.shape(y_pred)[0], tf.float32)))
        diversity_constraint = self.diversity_weight * tf.reduce_sum(y_pred * x_noisy)

        return self.accuracy_weight * tf.math.square(tf.norm((y_true - y_pred) * x_mask)) + (novelty_constraint + diversity_constraint)


    @staticmethod
    def prepare_train_data(ratings, mask):
        noise = 0.2
        flips = (tf.random.uniform(tf.shape(ratings)) > noise)
        noisy_ratings = ratings & flips
        return (ratings, mask, noisy_ratings), ratings


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.map(self.prepare_train_data)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        self.model.fit(dataset, epochs=self.epochs, steps_per_epoch=nr_records//self.batch_size)

    def save(self, path):
        self.model.save(path + '/constraint_auto_rec.h5')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def test(self, data: np.ndarray) -> np.ndarray:
        pass
