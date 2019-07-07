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
        self.name = kwargs.get('name', 'ConstraintAutoRec')
        self.optimizer = kwargs.get('optimizer', 'adam')

        self.params = kwargs

        self.prepare_model()

    def prepare_model(self):
        rating = Input(shape=(self.input_dim,), name='input_rating')

        mask = Input(shape=(self.input_dim,), name='input_mask')
        # y_mask = Input(shape=(self.input_dim,), name='input_y_mask')

        x = rating
        x = Dense(64, activation='relu')(x)
        latent = Dense(self.latent_dims, name='latent_vector', activation='relu')(x)

        self.encoder = Model(rating, latent, name='encoder')
        # encoder.summary()

        latent_inputs = Input(shape=(self.latent_dims,), name='decoder_input')
        x = latent_inputs
        x = Dense(64, activation='relu')(x)

        outputs = Dense(self.input_dim, name='decoder_output', activation='sigmoid')(x)
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()

        def constraint_loss(y_true, y_pred):
            return self.augmented_loss(y_true, y_pred, mask, rating)

        self.model = Model(inputs=[rating, mask], outputs=self.decoder(self.encoder(rating)),
                           name='ConstraintAutoRec')
        self.model.compile(optimizer=self.optimizer, loss=constraint_loss, metrics=['accuracy'])


    def augmented_loss(self, y_true, y_pred, mask, x_noisy):
        # error_constraint = alpha * tf.reduce_sum(estimated * actual)
        num_ratings = tf.reduce_sum(mask)
        num_ratings = tf.where(tf.equal(num_ratings, 0), 1.0, num_ratings)
        supervised_loss =  self.accuracy_weight * tf.math.square(tf.norm((y_true - y_pred) * mask)) / num_ratings
        novelty_constraint = self.novelty_weight * tf.reduce_sum(y_pred * (tf.reduce_sum(y_pred, 0) / tf.cast(tf.shape(y_pred)[0], tf.float32))) / tf.cast(tf.size(y_pred), tf.dtypes.float32)
        diversity_constraint = self.diversity_weight * tf.reduce_sum(y_pred * x_noisy) / tf.cast(tf.size(y_pred), tf.dtypes.float32)

        return supervised_loss + novelty_constraint + diversity_constraint




    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        self.model.fit(dataset, epochs=self.epochs, steps_per_epoch=nr_records//self.batch_size)

    def save(self, path):
        self.model.save(path + '/constraint_auto_rec.h5')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, data: np.ndarray) -> np.ndarray:
        mask_dummy = np.ones(data.shape)
        return self.model.predict((data, mask_dummy))

    def get_name(self) -> str:
        return self.name

    def get_params(self) -> dict:
        param_names = ['dimensions', 'latent_dims', 'accuracy_weight', 'novelty_weight', 'diversity_weight', 'epochs', 'batch_size', 'optimizer']
        return  {param: (self.__dict__[param]) for param in param_names}
