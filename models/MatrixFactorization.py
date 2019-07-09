import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel

class MF(tf.keras.Model):
    def __init__(self, nr_users, nr_items, latent_dim):
        super(tf.keras.Model, self).__init__()

        self.U = tf.Variable(initial_value=tf.random.truncated_normal([nr_users, latent_dim], name='latent_users'))
        self.P = tf.Variable(initial_value=tf.random.truncated_normal([latent_dim, nr_items], name='latent_items'))

    def call(self, inputs=None):
        return tf.matmul(self.U, self.P)


def loss(model, targets):
    predictions = model(None)
    user_indices = targets['user_id']
    specific = tf.gather(predictions, user_indices)
    target = tf.cast(targets['x'], tf.float32)
    return tf.reduce_mean(tf.square((specific - target) * tf.cast(targets['mask'], tf.float32)))

def grad(model, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, targets)
    return loss_value, tape.gradient(loss_value, [model.U, model.P])

class MatrixFactorization(BaseModel):


    def __init__(self, num_users, num_items, **kwargs):
        super().__init__(num_items, **kwargs)
        self.nr_users = num_users
        self.nr_items = num_items
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.epochs = kwargs.get('epochs', 10)
        self.model = MF(num_users, num_items, self.latent_dim)
        self.batch_size = kwargs.get('batch_size', 64)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)



    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.shuffle(2048)
        for i in range(self.epochs):
            step = 0
            for data in dataset:
                loss_value, grads = grad(self.model, data)
                self.optimizer.apply_gradients(zip(grads, [self.model.U, self.model.P]))
                if step % 10 == 0:
                    print("Epoch {} Loss at step {}: {:.3f}".format(i, step, loss_value))
                step += 1



    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        output = self.model(None).numpy()
        return output[user_ids]

    def get_name(self) -> str:
        pass

    def get_params(self) -> dict:
        param_names = ['latent_dim', 'epochs', 'batch_size', ]
        return  {param: (self.__dict__[param]) for param in param_names}
