import numpy as np
import tensorflow as tf

from models.BaseModel import BaseModel
import time


class NeuralLogicRec(tf.keras.Model):
    def __init__(self, nr_users, nr_items, embedding_dim):
        super(tf.keras.Model, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.Variable(initial_value=tf.random.normal([nr_users, embedding_dim]), name='embedding_user')
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, embedding_dim]), name='embedding_item')

        self.likes_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='likes_estimator')

        self.rated_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='rated_estimator')

        self.popular_estimator = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')], name='popular_estimator')

        self.nr_users = nr_users
        self.nr_items = nr_items

        #self.constraint_weights = tf.Variable(initial_value=tf.random.normal([4]), name='constraint_weights')
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
        estimated_rated = tf.squeeze(self.rated_estimator(input), axis=-1)

        popular = tf.squeeze(self.popular_estimator(self.item_embedding), axis=-1)

        return {'likes': estimated_likes, 'sim': self.calc_user_sim(embed_user, len(users)), 'rated': estimated_rated, 'popular': popular}

def sim(a, b):
    cos_similarity = tf.keras.losses.cosine_similarity(a, b,axis=-1)
    return ( 1 + cos_similarity) / 2

def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)

def Implies(a, b):
    return tf.minimum(1., 1 - a + b)

def Not(wff):
    return 1 - wff

def Forall(wff, axis=None):
    return tf.reduce_mean(wff, axis=axis)

def And(a, b):
    return tf.maximum(0.0, a + b - 1)


def combine_constraints(*constraints):
    def normalize_shape(tensor):
        if len(tensor.shape) == 1:
            return tensor
        elif len(tensor.shape) == 0:
            return tf.expand_dims(tensor, axis=0)
        else:
            return tf.reshape(tensor, [-1])
    return tf.concat([normalize_shape(x) for x in constraints], axis=0)


def constraint_satisfaction(model, likes, sim, rated, popular):

    # Rec(u, m) => Likes(u,m)
    # wff1 = tf.expand_dims(Forall(Implies(rec, likes)), axis=0)
    # Likes(u, m) => ~Rec(m)
    # wff2 = tf.expand_dims(Forall(Implies(likes, Not(rec))), axis=0)


    sim = tf.tile(tf.expand_dims(sim, axis=-1), [1, 1, likes.shape[1]])
    likes_u1 = tf.tile(tf.expand_dims(likes, axis=0), [sim.shape[0], 1, 1])
    likes_u2 = tf.tile(tf.expand_dims(likes, axis=1), [1, sim.shape[0], 1])
    # Sim(u1, u2) & Likes(u1, m) => likes(u2, m)
    wff3 = Forall(Implies(And(sim, likes_u1), likes_u2))
    # Sim(u1, u2) & ~Likes(u1, m) => ~likes(u2,m)
    wff4 = Forall(Implies(And(sim, Not(likes_u1)), Not(likes_u2)))
    # ~Sim(u1,u2) & Likes(u1,m) => ~likes(u2,m)
    wff5 = Forall(Implies(And(Not(sim), likes_u1), Not(likes_u2)))

    # Likes => Rated
    wff6 = Forall(Implies(likes, rated))

    popular_rep = tf.tile(tf.expand_dims(popular, axis=0), [likes.shape[0], 1])
    # Rated(u,m) => Popular(m)
    wff7 = Forall(Implies(rated, popular_rep))
    # Popular(m) => ~Likes(u,m)
    wff8 = Forall(Implies(popular_rep, Not(likes)))


    res = combine_constraints(wff3, wff4, wff5, wff6, wff7, wff8)
    return res


def supervised_target_loss(target, fnn):
    num_ratings = tf.reduce_sum(target['rated'])
    num_ratings = tf.where(tf.equal(num_ratings, 0), 1.0, num_ratings)
    likes_loss =  tf.math.square(tf.norm((target['likes'] - fnn['likes']) * target['rated'])) / num_ratings
    rated_loss = tf.keras.losses.mean_squared_error(target['rated'], fnn['rated'])
    return likes_loss + rated_loss

def ltn_loss(model, target, fnn):
    regularization = 0.0001 * tf.linalg.norm(model.user_embedding) + 0.0001 * tf.linalg.norm(model.item_embedding) + 0.001 # * tf.linalg.norm(model.constraint_weights)
    cost = regularization + supervised_target_loss(target, fnn)
    cost += (tf.reduce_sum(tf.abs((1 - constraint_satisfaction(model, fnn['likes'], fnn['sim'], fnn['rated'], fnn['popular'])))))
    return cost

def train_dtn(model, input):
    users = input['user_id']
    rated = tf.cast(input['x'], tf.float32)

    embed_user = tf.nn.embedding_lookup(model.user_embedding, users)

    target = {'likes': rated, 'rated': tf.cast(input['mask'], tf.float32) }
    with tf.GradientTape() as tape:
        fnn = model(users)
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
