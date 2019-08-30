import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel
from utils.common import printProgressBar


class MF(tf.keras.Model):
    def __init__(self, nr_users, nr_items, latent_dim):
        super(tf.keras.Model, self).__init__()

        self.U = tf.Variable(initial_value=tf.random.truncated_normal([nr_users, latent_dim], name='latent_users',mean=0.0,stddev=0.5))
        self.P = tf.Variable(initial_value=tf.random.truncated_normal([nr_items, latent_dim], name='latent_items',mean=0.0,stddev=0.5))
        self.regularization = 0.0001

    def call(self, user=None):
        specific = tf.nn.embedding_lookup(self.U, user)
        return tf.matmul(specific, self.P, transpose_b=True)


def loss(model, targets):
    user_indices = targets['user_id']
    pos = targets['pos']
    neg = targets['neg']


    embed_user = tf.nn.embedding_lookup(model.U, user_indices)
    embed_pos = tf.nn.embedding_lookup(model.P, pos)
    embed_neg = tf.nn.embedding_lookup(model.P, neg)

    pos_score = tf.matmul(embed_user, embed_pos, transpose_b=True)
    neg_score = tf.matmul(embed_user, embed_neg, transpose_b=True)
    # print(pos_score, neg_score)
    reg_term = + model.regularization * (tf.math.square(tf.norm(model.U)) + tf.math.square(tf.norm(model.P)))
    return tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_score - neg_score))) + reg_term 

    # predictions = model(user_indices)
    # target = tf.cast(targets['x'], tf.float32)
    # return tf.reduce_mean(tf.square((predictions - target)))

def grad(model, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, targets)
    return loss_value, tape.gradient(loss_value, [model.U, model.P])


class BPR(BaseModel):


    def __init__(self, num_users, num_items, **kwargs):
        super().__init__(num_items, **kwargs)
        self.nr_users = num_users
        self.nr_items = num_items
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.epochs = kwargs.get('epochs', 10)
        self.model = MF(num_users, num_items, self.latent_dim)
        self.batch_size = kwargs.get('batch_size', 128)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

    @staticmethod
    def transform_train_data(record):
        positive = record['x']
        cond = tf.equal(positive, True)
        pos =  tf.where(cond)
        nr_pos = tf.shape(pos)
        neg = tf.random.shuffle(tf.where(tf.math.logical_not(cond)))[:nr_pos[0]]
        user_id = tf.fill(nr_pos, record['user_id'])
        return tf.data.Dataset.from_tensor_slices({'pos': pos, 'neg':neg, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.flat_map(self.transform_train_data).batch(self.batch_size)
        dataset = dataset.shuffle(2048)
        nr_steps = nr_records // self.batch_size
        for i in range(self.epochs):
            step = 0
            for data in dataset:
                loss_value, grads = grad(self.model, data)
                self.optimizer.apply_gradients(zip(grads, [self.model.U, self.model.P]))
                # printProgressBar(step, nr_steps, 'Epoch {}, loss:  {:.3f}'.format(i, loss_value),length=80)
                if step % 10 == 0:
                    print("\rEpoch #{} Loss at step {}: {:.4f}".format(i, step, loss_value), end='\r')
                step += 1
            print()


    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        return self.model(user_ids).numpy()

    def get_name(self) -> str:
        return "BPR"

    def get_params(self) -> dict:
        param_names = ['latent_dim', 'epochs', 'batch_size', ]
        return  {param: (self.__dict__[param]) for param in param_names}
