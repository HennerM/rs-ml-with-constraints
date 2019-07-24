import numpy as np
import tensorflow as tf
import itertools

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
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, 24]), name='embedding_item')

        self.likes_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='likes_estimator')

        self.rec_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='rec_estimator')

        self.nr_users = nr_users
        self.nr_items = nr_items

        self.constraint_weights = tf.Variable(initial_value=tf.random.normal([4]), name='constraint_weights')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


    def call(self, users, items):
        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_item = tf.nn.embedding_lookup(self.item_embedding, items)

        input = tf.concat([embed_user, embed_item], axis=1)
        estimated_likes = tf.squeeze(self.likes_estimator(input))
        estimated_rec = tf.squeeze(self.rec_estimator(input))
        output = {'likes': estimated_likes, 'rec': estimated_rec}

        return output


def sim(a, b):
    cos_similarity = tf.keras.losses.cosine_similarity(a, b,axis=1)
    return ( 1 + cos_similarity) / 2

def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)

def negative_mse(network_output, y):
    return tf.math.negative(supervised_loss(network_output['likes'], y['likes']))


def Implies(a, b):
    return tf.minimum(1., 1 - a + b)

def Not(wff):
    return 1 - wff

def Forall(wff):
    return tf.reduce_mean(wff, axis=0)

def And(a, b):
    return tf.maximum(0.0, a + b - 1)

def constraint_satisfaction(model, y):
    # Likes(u, m) => ~Rec(m)
    wff1 = Forall(Implies(y['rec'], y['likes']))
    wff2 = Forall(Implies(y['likes'], Not(y['rec'])))
    # Sim(u1, u2) & Likes(u1, m) => Rec(u2, m)
    # TODO

    user_cross = tf.convert_to_tensor([x for x in itertools.permutations(range(10), 2)])
    u1 = user_cross[:,0]
    u2 = user_cross[:,1]
    embed_u1 = tf.nn.embedding_lookup(model.user_embedding, u1)
    embed_u2 = tf.nn.embedding_lookup(model.user_embedding, u2)
    cos_sim = sim(embed_u1, embed_u2)
    wff3 = tf.zeros([10])
    for i in range(1):
        item = tf.broadcast_to(tf.nn.embedding_lookup(model.item_embedding, [i]), [embed_u1.shape[0], 24])

        likes = tf.squeeze(model.likes_estimator(tf.concat([embed_u1, item], axis=1)))
        rec = tf.squeeze(model.rec_estimator(tf.concat([embed_u2, item], axis=1)))
        wff3 = Forall(Implies(And(cos_sim, likes), rec))
        wff4 = Forall(Implies(And(cos_sim, Not(likes)), Not(rec)))

    return tf.stack([wff1, wff2, wff3, wff4], axis=0)

def map_inference(model, network_output, convergence_e = 1e-3):
    y = {"likes": tf.Variable(initial_value=tf.random.truncated_normal(network_output['likes'].shape, mean=.5,stddev=.25), constraint=lambda t: tf.clip_by_value(t, 0., 1.)),
            "rec": tf.Variable(initial_value=tf.random.truncated_normal(network_output['rec'].shape, mean=.5, stddev=.25), constraint=lambda t: tf.clip_by_value(t, 0., 1.)),
         }

    previous_loss = None
    for i in range(2048):
        with tf.GradientTape() as tape:
            l = tf.math.negative(negative_mse(network_output, y) + tf.reduce_sum(model.constraint_weights * constraint_satisfaction(model, y)))
        grad = tape.gradient(l, [y['likes'], y['rec']])
        model.optimizer.apply_gradients(zip(grad, [y['likes'], y['rec'] ]))
        if previous_loss is not None and tf.abs(previous_loss - l) < convergence_e:
            break
        previous_loss = l

    return y

def supervised_target_loss(target, fnn):
    return supervised_loss(fnn['likes'], target['likes'])

def supervised_map_loss(map, fnn):
    return supervised_loss(fnn['likes'], map['likes']) + supervised_loss(fnn['rec'], map['rec'])

def ltn_loss(model, target, map_solution, fnn):
    regularization = 0.0001 * tf.linalg.norm(model.user_embedding) + 0.0001 * tf.linalg.norm(model.item_embedding) + 0.001 * tf.linalg.norm(model.constraint_weights)
    cost = regularization + supervised_target_loss(target, fnn)/2 + supervised_map_loss(map_solution, fnn)/2
    cost += tf.reduce_sum(model.constraint_weights * (constraint_satisfaction(model, target) - constraint_satisfaction(model, map_solution)))
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    @staticmethod
    def transform_train_data(record):
        rated = tf.squeeze(tf.where(record['mask']))

        nr_rated = tf.shape(rated)
        user_id = tf.squeeze(tf.fill(nr_rated, record['user_id']))
        rating = tf.cast(tf.squeeze(tf.boolean_mask(record['x'], record['mask'])), tf.dtypes.float32)
        return tf.data.Dataset.from_tensor_slices({'item_id': rated, 'rating': rating, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.shuffle(64)
        dataset = dataset.flat_map(self.transform_train_data).batch(self.batch_size)
        for i in range(self.epochs):
            step = 0
            epcoh_start = time.time()
            for data in dataset:
                # loss_value, grads = grad(self.model, data)
                loss_value, grads = train_dtn(self.model, data, {'likes': data['rating'], 'rec': tf.zeros_like(data['rating'])})
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                # printProgressBar(step, nr_steps, 'Epoch {}, loss:  {:.3f}'.format(i, loss_value),length=80)
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
        return inference['rec'].numpy().T

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        user_ids = tf.convert_to_tensor(user_ids)
        items = tf.convert_to_tensor(np.arange(self.nr_items))

        users = tf.tile(user_ids, items.shape)
        items = tf.sort(tf.tile(items, user_ids.shape))
        predictions = self.model(users, items)
        inference = map_inference(self.model, predictions)
        tmp = tf.reshape(inference['rec'], [self.nr_items, len(user_ids)])
        return tf.transpose(tmp).numpy()

    def get_name(self) -> str:
        return "NeuralLogicRec"

    def get_params(self) -> dict:
        items = np.arange(self.nr_items)
        param_names = ['latent_dim', 'epochs', 'batch_size', ]
        return  {param: (self.__dict__[param]) for param in param_names}
