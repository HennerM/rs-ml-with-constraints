from models.BaseModel import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import Model, backend as K


class ConstraintAutoRec(BaseModel):

    def __init__(self, nr_users, nr_items, **kwargs):
        super().__init__(nr_users, nr_items, **kwargs)
        self.prepare_model()
        self.latent_dims = kwargs.get('latent_dim', 64)
        self.input_dim = nr_items
        self.accuracy_weight = kwargs.get('accuracy_weight', 1.0)
        self.novelty_weight = kwargs.get('novelty_weight', 0.5)
        self.diversity_weight = kwargs.get('diversity_weight', 0.5)
        self.epochs = kwargs.get('epochs', 20)

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
            self.augmented_loss(y_true, y_pred, mask, rating_noisy)

        self.model = Model(inputs=[rating, mask, rating_noisy], outputs=self.decoder(self.encoder(rating_noisy)),
                           name='ConstraintAutoRec')
        self.model.compile(optimizer='adam', loss=constraint_loss, metrics=['accuracy'])


    def augmented_loss(self, y_true, y_pred, x_mask, x_noisy):
        # error_constraint = alpha * tf.reduce_sum(estimated * actual)
        novelty_constraint = self.novelty_weight * tf.reduce_sum(y_pred * (tf.reduce_sum(y_pred, 0) / tf.cast(tf.shape(y_pred)[0], tf.float32)))
        diversity_constraint = self.diversity_weight * tf.reduce_sum(y_pred * x_noisy)

        return self.accuracy_weight * tf.math.square(tf.norm((y_true - y_pred) * x_mask)) + (novelty_constraint + diversity_constraint)


    @staticmethod
    def prepare_train_data(record):
        noise = 0.2
        ratings = record[0]
        flips = (tf.random.uniform(record.shape) > noise)
        noisy_ratings = ratings * flips
        # TODO test if this works
        return (record[0], record[1], noisy_ratings), record[0]


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.map(self.prepare_train_data)
        dataset = dataset.batch(32)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)
        self.model.fit(dataset, epochs=20, steps_per_epoch=19)
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def test(self, input_data):
        pass
