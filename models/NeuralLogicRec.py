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

        self.user_embedding = tf.Variable(initial_value=tf.random.normal([nr_users, embedding_dim],  name='embedding_user'))
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, embedding_dim], name='embedding_item'))

        self.rec_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')])
        self.nr_users = nr_users
        self.nr_items = nr_items


    def call(self, users, items):
        batch_size = tf.shape(users)[0]

        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_item = tf.nn.embedding_lookup(self.item_embedding, items)

        input = tf.concat([embed_user, embed_item], axis=1)
        output = self.rec_estimator(input)

        return output



def loss_from_input(model, targets):
    users = targets['user_id']
    items = targets['item_id']
    rating = tf.cast(targets['rating'], tf.float32)
    predictions = model(users, items)
    return supervised_loss(predictions, rating)  + 0.001 * tf.linalg.norm(model.user_embedding) + 0.001 * tf.linalg.norm(model.item_embedding)

def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)


def grad(model, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_from_input(model, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def constraint_loss(y):

    # Likes(u, m) => ~Rec(m)
    # Likes(u,
    return 0.0

def map_inference(model, network_output):
    y = tf.Variable(initial_value=tf.random.truncated_normal([network_output.shape]))
    for i in range(10):
        with tf.GradientTape() as tape:
            l = supervised_loss(network_output, y) + constraint_loss(y)
        grad = tape.gradient(l, [y])
        model.optimzer.apply_gradients(zip(grad, y))

    return y


def ltn_loss(model, target, map_solution, fnn, input):
    regularization = tf.linalg.norm(model.trainable_variables)/2 + tf.linalg.norm(model.constraint_params)/2
    cost = regularization + tf.keras.losses.mean_squared_error(target, fnn)/2 + tf.keras.losses.mean_squared_error(map_solution, fnn)/2
    for c in model.constraints:
        cost += c.weight * (constraint_loss(target) - constraint_loss(map_solution))
    return cost

def train_dtn(model, input):
    X = input
    for i in range(100):
        fnn = model(X)
        map_solution = map_inference(model, fnn)




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
        rating = tf.squeeze(tf.boolean_mask(record['x'], record['mask']))
        return tf.data.Dataset.from_tensor_slices({'item_id': rated, 'rating': rating, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.shuffle(256).flat_map(self.transform_train_data).batch(self.batch_size)
        dataset = dataset.shuffle(10_000)
        nr_steps = nr_records // self.batch_size
        for i in range(self.epochs):
            step = 0
            epcoh_start = time.time()
            for data in dataset:
                loss_value, grads = grad(self.model, data)
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
        predictions = self.model(user, items).numpy()
        return predictions.T

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
