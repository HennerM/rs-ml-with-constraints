import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel
from utils.common import printProgressBar
import time


class NeuralLogicRec(tf.keras.Model):
    def __init__(self, nr_users, nr_items, embedding_dim):
        super(tf.keras.Model, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_transform = tf.keras.layers.Dense(units=self.embedding_dim, activation='relu')
        self.item_transform = tf.keras.layers.Dense(units=self.embedding_dim, activation='relu')

        self.user_embedding = tf.Variable(initial_value=tf.random.normal([nr_users, embedding_dim]), name='embedding_user')
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, embedding_dim]), name='embedding_item')

        self.likes_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='likes_estimator')

        # self.rec_estimator = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=32, activation='relu'),
        #     tf.keras.layers.Dense(units=16, activation='relu'),
        #     tf.keras.layers.Dense(units=1, activation='sigmoid')], name='rec_estimator')

        self.nr_users = nr_users
        self.nr_items = nr_items

        self.constraint_weights = tf.Variable(initial_value=tf.random.normal([1]), name='constraint_weights')
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


    def call(self, users, items):
        batch_size = tf.shape(users)[0]

        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_item = tf.nn.embedding_lookup(self.item_embedding, items)

        input = tf.concat([embed_user, embed_item], axis=1)
        output = {'likes': self.likes_estimator(input), 'rec': self.likes_estimator(input) }

        return output



def loss_from_input(model, targets):
    users = targets['user_id']
    items = targets['item_id']
    rating = tf.cast(targets['rating'], tf.float32)
    predictions = model(users, items)
    return supervised_loss(predictions, rating)  + 0.001 * tf.linalg.norm(model.user_embedding) + 0.001 * tf.linalg.norm(model.item_embedding)

def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)

def nn_loss(network_output, y):
    return tf.reduce_mean(supervised_loss(network_output['likes'], y['likes']))

def grad(model, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_from_input(model, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def Not(wff):
    return 1 - wff

def Forall(wff):
    return tf.reduce_mean(wff)

def constraint_satisfaction(y):
    # Likes(u, m) => ~Rec(m): 1 - tf.abs(Likes(u,m), 1 - Rec(u, m)
    norm = 1 - tf.abs(tf.clip_by_value(y['likes'],0.,1.0) - Not(tf.clip_by_value(y['rec'], 0., 1.0)))
    return Forall(norm)

def map_inference(model, network_output):
    y = {"likes": tf.Variable(initial_value=tf.random.truncated_normal(network_output['likes'].shape, mean=.5,stddev=.25)),
            "rec": tf.Variable(initial_value=tf.random.truncated_normal(network_output['rec'].shape, mean=.5, stddev=.25)),
         }

    previous_loss = None
    for i in range(16):
        with tf.GradientTape() as tape:
            l = tf.math.negative(nn_loss(network_output, y) + model.constraint_weights * constraint_satisfaction(y))
        grad = tape.gradient(l, [y['likes'], y['rec']])
        model.optimizer.apply_gradients(zip(grad, [y['likes'], y['rec'] ]))
        if previous_loss is not None and tf.abs(previous_loss - l) < 1e-5:
            # print('converged at iteration {}'.format(i))
            break
        # print(l)
        previous_loss = l

    return y


def ltn_loss(model, target, map_solution, fnn):
    regularization = 0
    cost = regularization + tf.linalg.norm(target['likes'] - fnn['likes'])/2 + tf.linalg.norm(map_solution['likes'] - fnn['likes'])/2
    cost += model.constraint_weights * (constraint_satisfaction(target) - constraint_satisfaction(map_solution))
    return cost

def train_dtn(model, input, target):
    with tf.GradientTape() as tape:
        fnn = model(input['user_id'], input['item_id'])
        map_solution = map_inference(model, fnn)
        loss = ltn_loss(model, target, map_solution, fnn)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads

class NLR(BaseModel):


    def __init__(self, num_users, num_items, **kwargs):
        super().__init__(num_items, **kwargs)
        self.nr_users = num_users
        self.nr_items = num_items
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.embedding_dim = kwargs.get('embedding_dim', 128)
        self.epochs = kwargs.get('epochs', 10)
        self.model = NeuralLogicRec(num_users, num_items, self.embedding_dim)
        self.batch_size = kwargs.get('batch_size', 128)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    @staticmethod
    def transform_train_data(record):
        rated = tf.squeeze(tf.where(record['mask']))

        nr_rated = tf.shape(rated)
        user_id = tf.squeeze(tf.fill(nr_rated, record['user_id']))
        rating = tf.cast(tf.squeeze(tf.boolean_mask(record['x'], record['mask'])), tf.dtypes.float32)
        return tf.data.Dataset.from_tensor_slices({'item_id': rated, 'rating': rating, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        # dataset = dataset.shuffle(512)
        dataset = dataset.flat_map(self.transform_train_data).batch(self.batch_size)
        dataset = dataset #.shuffle(20_000)
        for i in range(self.epochs):
            step = 0
            epcoh_start = time.time()
            for data in dataset:
                # loss_value, grads = grad(self.model, data)
                loss_value, grads = train_dtn(self.model, data, {'likes': data['rating'], 'rec': tf.zeros_like(data['rating'])})
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                # printProgressBar(step, nr_steps, 'Epoch {}, loss:  {:.3f}'.format(i, loss_value),length=80)
                if step % 10 == 0:
                    diff = time.time() - epcoh_start
                    print("\rEpoch #{} Loss at step {}: {:.4f}, time: {:.3f}".format(i, step, tf.reduce_mean(loss_value).numpy(), diff), end='\r')
                step += 1
            print()


    def save(self, path):
        pass

    def load(self, path):
        pass


    def predict_single_user(self, user):

        items = np.arange(self.nr_items)
        user = np.repeat(user, self.nr_items)
        predictions = self.model(user, items)
        inference = map_inference(self.model, predictions)
        return predictions['likes'].numpy().T

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        # output =  np.asarray(list(map(lambda x: self.predict_single_user(x), user_ids)))
        output = np.zeros(data.shape)
        for user_id in range(data.shape[0]):
            output[user_id] = self.predict_single_user(user_id)
        return output

    def get_name(self) -> str:
        return "NeuralLogicRec"

    def get_params(self) -> dict:
        items = np.arange(self.nr_items)
        param_names = ['latent_dim', 'epochs', 'batch_size', ]
        return  {param: (self.__dict__[param]) for param in param_names}
