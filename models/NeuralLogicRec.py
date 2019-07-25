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
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, embedding_dim]), name='embedding_item')

        self.likes_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='likes_estimator')

        self.rec_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=16, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='rec_estimator')

        self.nr_users = nr_users
        self.nr_items = nr_items

        # self.constraint_weights = tf.Variable(initial_value=tf.random.normal([4]), name='constraint_weights', constraint=lambda t: tf.clip_by_value(t, 0., 1.))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)



    def calc_user_sim(self, embed_user, nr_users):
        users_1 = tf.tile(tf.expand_dims(embed_user, axis=1), [1, nr_users, 1])
        users_2 = tf.tile(tf.expand_dims(embed_user, axis=0), [nr_users,1, 1])
        return sim(users_1, users_2)

    def call(self, users):
        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_user_likes = tf.tile(tf.expand_dims(embed_user, axis=1), [1, self.nr_items, 1])
        expanded_embed = tf.expand_dims(self.item_embedding, axis=0)
        embed_item = tf.tile(expanded_embed, [len(users), 1, 1])
        input = tf.concat([embed_user_likes, embed_item], axis=-1)
        estimated_likes = tf.squeeze(self.likes_estimator(input), axis=-1)
        estimated_rec = tf.squeeze(self.rec_estimator(input), axis=-1)

        return {'likes': estimated_likes, 'rec': estimated_rec, 'sim': self.calc_user_sim(embed_user, len(users))}

    def train(self, user_cross):
        embed_user = tf.expand_dims(tf.nn.embedding_lookup(self.user_embedding, user_cross), axis=2)
        embed_user = tf.tile(embed_user, [1, 1, self.nr_items, 1])
        expanded_embedd = tf.reshape(self.item_embedding, [1, 1, self.nr_items, self.embedding_dim])
        embed_item = tf.tile(expanded_embedd, [len(user_cross), 2, 1, 1])


        input = tf.concat([embed_user, embed_item], axis=-1)
        estimated_likes = tf.squeeze(self.likes_estimator(input))
        estimated_rec = tf.squeeze(self.rec_estimator(input))
        output = {'likes': estimated_likes, 'rec': estimated_rec}

        return output


def sim(a, b):
    cos_similarity = tf.keras.losses.cosine_similarity(a, b,axis=-1)
    return ( 1 + cos_similarity) / 2

def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)

def negative_mse(network_output, y_likes, y_rec):
    return tf.reduce_sum(tf.math.negative(supervised_loss(network_output['likes'], y_likes) + supervised_loss(network_output['rec'], y_rec))) # TODO add rec loss


def Implies(a, b):
    return tf.minimum(1., 1 - a + b)

def Not(wff):
    return 1 - wff

def Forall(wff, axis=None):
    return tf.reduce_mean(wff, axis=axis)

def And(a, b):
    return tf.maximum(0.0, a + b - 1)

def constraint_satisfaction(model, likes, rec, sim):

    # Rec(u, m) => Likes(u,m)
    wff1 = tf.expand_dims(Forall(Implies(rec, likes)), axis=0)
    # Likes(u, m) => ~Rec(m)
    wff2 = tf.expand_dims(Forall(Implies(likes, Not(rec))), axis=0)

    # Sim(u1, u2) & Likes(u1, m) => Rec(u2, m)
    sim = tf.tile(tf.expand_dims(sim, axis=-1), [1, 1, likes.shape[1]])
    likes = tf.tile(tf.expand_dims(likes, axis=0), [sim.shape[0], 1, 1])
    rec = tf.tile(tf.expand_dims(rec, axis=1), [1, sim.shape[0], 1])
    # rec2 = tf.tile(tf.expand_dims(rec, axis=0), [sim.shape[0], 1, 1])

    wff3 = tf.expand_dims(Forall(Implies(And(sim, likes), rec)), axis=0)
    wff4 = tf.expand_dims(Forall(Implies(And(sim, Not(likes)), Not(rec))), axis=0)
    # tmp = Forall(Implies(And(sim, likes), rec2), axis=[0,1])
    # print(wff3.numpy().mean(), tmp.numpy().mean())

    res = tf.concat([wff1, wff2, wff3, wff4], axis=0)

    return res

def map_inference(model, network_output, convergence_e = 1e-3):
    convergence_e = tf.convert_to_tensor(convergence_e)
    y_likes =  tf.Variable(initial_value=tf.random.truncated_normal(network_output['likes'].shape, mean=.5,stddev=.25), constraint=lambda t: tf.clip_by_value(t, 0., 1.))
    y_recs = tf.Variable(initial_value=tf.random.truncated_normal(network_output['rec'].shape, mean=.5, stddev=.25), constraint=lambda t: tf.clip_by_value(t, 0., 1.))
    y_sim = tf.Variable(initial_value=tf.random.truncated_normal(network_output['sim'].shape, mean=.5, stddev=.25), constraint=lambda t: tf.clip_by_value(t, 0., 1.))

    previous_loss = None
    for i in range(2048):
        with tf.GradientTape() as tape:
            l = tf.math.negative(negative_mse(network_output, y_likes, y_recs) + tf.reduce_sum(model.constraint_weights * constraint_satisfaction(model, y_likes, y_recs, y_sim)))
        grad = tape.gradient(l, [y_likes, y_recs])
        model.optimizer.apply_gradients(zip(grad, [y_likes, y_recs]))
        # if i % 10 == 0:
            # print("MAP Loss at step {}: {:.4f}".format(i, l.numpy()), end='\r')
        if previous_loss is not None and tf.abs(previous_loss - l) < convergence_e:
            break
        previous_loss = l

    return {'likes': y_likes, 'rec': y_recs, 'sim': y_sim}

def supervised_target_loss(target, fnn):
    return supervised_loss(fnn['likes'], target['likes'])

def supervised_map_loss(map, fnn):
    return supervised_loss(fnn['likes'], map['likes']) + supervised_loss(fnn['rec'], map['rec'])

def ltn_loss(model, target, fnn):
    regularization = 0.0001 * tf.linalg.norm(model.user_embedding) + 0.0001 * tf.linalg.norm(model.item_embedding) + 0.001 # * tf.linalg.norm(model.constraint_weights)
    cost = regularization + supervised_target_loss(target, fnn) / 2 # + supervised_map_loss(map, fnn) / 2
    cost += (tf.reduce_sum((1 - constraint_satisfaction(model, fnn['likes'], fnn['rec'], fnn['sim']))))
    return cost

def train_dtn(model, input):
    users = input['user_id']
    rated = tf.cast(input['x'], tf.float32)

    embed_user = tf.nn.embedding_lookup(model.user_embedding, users)

    target = {'likes': rated }
    # user_cross = tf.convert_to_tensor([x for x in itertools.permutations(users.numpy(), 2)])
    with tf.GradientTape() as tape:
        fnn = model(users)
        # map_solution = map_inference(model, fnn)
        loss = ltn_loss(model, target, fnn)
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @staticmethod
    def transform_train_data(record):
        rated = tf.squeeze(tf.where(record['mask']))

        nr_rated = tf.shape(rated)
        user_id = tf.squeeze(tf.fill(nr_rated, record['user_id']))
        rating = tf.cast(tf.squeeze(tf.boolean_mask(record['x'], record['mask'])), tf.dtypes.float32)
        return tf.data.Dataset.from_tensor_slices({'item_id': rated, 'rating': rating, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.shuffle(64)
        # dataset = dataset.flat_map(self.transform_train_data)
        dataset = dataset.batch(self.batch_size)
        for i in range(self.epochs):
            step = 0
            epcoh_start = time.time()
            for data in dataset:
                # loss_value, grads = grad(self.model, data)
                loss_value, grads = train_dtn(self.model, data)
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
        predictions = self.model([user])
        # inference = map_inference(self.model, predictions)
        return predictions['rec'].numpy().T

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        user_ids = tf.convert_to_tensor(user_ids)
        predictions = self.model(user_ids)
        # inference = map_inference(self.model, predictions)
        tmp = tf.reshape(predictions['likes'], [self.nr_items, len(user_ids)])
        return tf.transpose(tmp).numpy()

    def get_name(self) -> str:
        return "NeuralLogicRec"

    def get_params(self) -> dict:
        items = np.arange(self.nr_items)
        param_names = ['latent_dim', 'epochs', 'batch_size', ]
        return  {param: (self.__dict__[param]) for param in param_names}
