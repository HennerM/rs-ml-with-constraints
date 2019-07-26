import numpy as np
import tensorflow as tf
from evaluation import Evaluation 

from models.BaseModel import BaseModel
import time


class NeuralLogicRec(tf.keras.Model):
    def __init__(self, nr_users, nr_items, embedding_dim, nr_hidden_layers, nr_item_samples):
        super(tf.keras.Model, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.Variable(initial_value=tf.random.normal([nr_users, embedding_dim]), name='embedding_user')
        self.item_embedding = tf.Variable(initial_value=tf.random.normal([nr_items, embedding_dim]), name='embedding_item') 
        
        self.likes_estimator = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=embedding_dim * 2, activation='relu') for i in range(nr_hidden_layers)] + 
            [tf.keras.layers.Dense(units=1, activation='sigmoid')], name='likes_estimator')

        self.rated_estimator = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=embedding_dim * 2, activation='relu') for i in range(nr_hidden_layers)] + 
            [tf.keras.layers.Dense(units=1, activation='sigmoid')], name='rated_estimator')

        self.popular_estimator = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=embedding_dim * 2, activation='relu') for i in range(nr_hidden_layers)] + 
            [tf.keras.layers.Dense(units=1, activation='sigmoid')], name='popular_estimator')

        self.nr_users = nr_users
        self.nr_items = nr_items
        self.nr_item_samples = nr_item_samples

    @tf.function
    def calc_embedding_sim(self, embed_a, embed_b):
        a = tf.tile(tf.expand_dims(embed_a, axis=1), [1, len(embed_b), 1])
        b = tf.tile(tf.expand_dims(embed_b, axis=0), [len(embed_a),1, 1])
        return sim(a, b)
    
    @tf.function
    def item_sim(self, items_a, items_b):
        a = tf.nn.embedding_lookup(self.item_embedding, items_a)
        b = tf.nn.embedding_lookup(self.item_embedding, items_b)
        return sim(a, b)

    def predict(self, users):
        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_user_likes = tf.tile(tf.expand_dims(embed_user, axis=1), [1, self.nr_items, 1])
        expanded_embed = tf.expand_dims(self.item_embedding, axis=0)
        embed_item = tf.tile(expanded_embed, [len(users), 1, 1])
        input = tf.concat([embed_user_likes, embed_item], axis=-1)
        return tf.squeeze(self.likes_estimator(input), axis=-1)

    def call(self, users):
        embed_user = tf.nn.embedding_lookup(self.user_embedding, users)
        embed_user_likes = tf.tile(tf.expand_dims(embed_user, axis=1), [1, self.nr_items, 1])
        expanded_embed = tf.expand_dims(self.item_embedding, axis=0)
        embed_item = tf.tile(expanded_embed, [len(users), 1, 1])
        input = tf.concat([embed_user_likes, embed_item], axis=-1)
        estimated_likes = tf.squeeze(self.likes_estimator(input), axis=-1)
        estimated_rated = tf.squeeze(self.rated_estimator(input), axis=-1)

        popular = tf.squeeze(self.popular_estimator(self.item_embedding), axis=-1)

        return {'likes': estimated_likes, 
                'sim': self.calc_embedding_sim(embed_user, embed_user),
                'rated': estimated_rated, 
                'popular': popular,
                 }

@tf.function
def sim(a, b):
    return tf.keras.losses.cosine_similarity(a, b,axis=-1)

@tf.function
def supervised_loss(predictions, target):
    return tf.keras.losses.mean_squared_error(target, predictions)

@tf.function
def Implies(a, b):
    return tf.minimum(1., 1 - a + b)

@tf.function
def Not(wff):
    return 1 - wff

@tf.function
def Forall(wff, axis=None):
    return tf.reduce_mean(wff, axis=axis)

@tf.function
def And(a, b):
    return tf.maximum(0.0, a + b - 1)

@tf.function
def combine_constraints(*constraints):
    def normalize_shape(tensor):
        if len(tensor.shape) == 1:
            return tensor
        elif len(tensor.shape) == 0:
            return tf.expand_dims(tensor, axis=0)
        else:
            return tf.reshape(tensor, [-1])
    return tf.concat([normalize_shape(x) for x in constraints], axis=0)

@tf.function
def item_cf(model, likes):
    
    item_sample_a = tf.random.uniform([model.nr_item_samples], minval=0, maxval=model.nr_items, dtype=tf.int32)
    item_sample_b = tf.random.uniform([model.nr_item_samples], minval=0, maxval=model.nr_items, dtype=tf.int32)
    item_sim = model.item_sim(item_sample_a, item_sample_b)
    sim = tf.maximum(0.0, item_sim)
    anti_sim = tf.abs(tf.minimum(0.0, item_sim))
    likes_1_sample = tf.gather(likes, item_sample_a, axis=1)
    likes_2_sample = tf.gather(likes, item_sample_b, axis=1)
    # (Likes(u, m1) & Sim(m1, m2) => Likes(u, m2)) & (Likes(u, m2) & Sim(m1, m2) => Likes(u, m1)
    a = Forall(And(
        Implies(And(likes_1_sample, sim), likes_2_sample),
        Implies(And(likes_2_sample, sim), likes_1_sample)
    ))
    # (~Likes(u, m1) & Sim(m1, m2) => ~Likes(u, m2)) & (~Likes(u, m2) & Sim(m1, m2) => ~Likes(u, m1))
    b = Forall(And(
        Implies(And(Not(likes_1_sample), sim), Not(likes_2_sample)),
        Implies(And(Not(likes_2_sample), sim), Not(likes_1_sample)),
    ))
    # (Likes(u, m1) & AntiSim(m1, m2) => ~Likes(u, m2)) & (Likes(u, m2) & AntiSim(m1, m2) => ~Likes(u, m1))
    c = Forall(And(
        Implies(And(likes_1_sample, anti_sim), Not(likes_2_sample)),
        Implies(And(likes_2_sample, anti_sim), Not(likes_1_sample))
    ))
    return tf.stack([a, b, c], axis=0)

@tf.function
def constraint_satisfaction(model, likes, cos_sim, rated, popular):

    cos_sim = tf.tile(tf.expand_dims(cos_sim, axis=-1), [1, 1, likes.shape[1]])
    
    sim = tf.maximum(0.0, cos_sim)
    anti_sim = tf.abs(tf.minimum(0.0, cos_sim))
    likes_u1 = tf.tile(tf.expand_dims(likes, axis=0), [sim.shape[0], 1, 1])
    likes_u2 = tf.tile(tf.expand_dims(likes, axis=1), [1, sim.shape[0], 1])
    # Sim(u1, u2) & Likes(u1, m) => likes(u2, m)
    wff1 = Forall(Implies(And(sim, likes_u1), likes_u2))
    # Sim(u1, u2) & ~Likes(u1, m) => ~likes(u2,m)
    wff2 = Forall(Implies(And(sim, Not(likes_u1)), Not(likes_u2)))
    # ~Sim(u1,u2) & Likes(u1,m) => ~likes(u2,m)
    wff3 = Forall(Implies(And(anti_sim, likes_u1), Not(likes_u2)))

    # Likes => Rated
    wff4 = Forall(Implies(likes, rated))

    popular_rep = tf.tile(tf.expand_dims(popular, axis=0), [likes.shape[0], 1])
    # Rated(u,m) => Popular(m)
    wff5 = Forall(Implies(rated, popular_rep))
    # ~Popular(m) => Likes(u,m)
    wff6 = Forall(Implies(Not(popular_rep), likes))
 
    wff7 = Forall(Implies(And(Not(rated), likes), likes))
    wff8 = Forall(Not(popular))
    
    wff9 = item_cf(model, likes)
    
    res = combine_constraints(wff1, wff2, wff3, wff4, wff5, wff6, wff7, wff8, wff9)
    return res

@tf.function
def supervised_target_loss(target, fnn):
    num_ratings = tf.reduce_sum(target['rated'])
    num_ratings = tf.where(tf.equal(num_ratings, 0), 1.0, num_ratings)
    likes_loss =  tf.math.square(tf.norm((target['likes'] - fnn['likes']) * target['rated'])) / num_ratings
    rated_loss = tf.keras.losses.mean_squared_error(target['rated'], fnn['rated'])
    return likes_loss + rated_loss

@tf.function
def ltn_loss(model, target, fnn):
    regularization = 0.0001 * tf.linalg.norm(model.user_embedding) + 0.0001 * tf.linalg.norm(model.item_embedding) + 0.001 # * tf.linalg.norm(model.constraint_weights)
    cost = regularization + supervised_target_loss(target, fnn)
    cost += (tf.reduce_sum((1 - constraint_satisfaction(model, fnn['likes'], fnn['sim'], fnn['rated'], fnn['popular']))))
    return cost

@tf.function
def train_dtn(model, users, rated, mask):
    
    target = {'likes': rated, 'rated': mask}
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
        self.embedding_dim = kwargs.get('embedding_dim', 16)
        self.nr_hidden_layers = kwargs.get('nr_hidden_layers', 4)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 16)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.epochs_trained = 0
        self.nr_item_samples = kwargs.get('nr_item_samples', 512)
        
        self.model = NeuralLogicRec(num_users, num_items, self.embedding_dim, self.nr_hidden_layers, self.nr_item_samples)

    @staticmethod
    def transform_train_data(record):
        rated = tf.squeeze(tf.where(record['mask']))

        nr_rated = tf.shape(rated)
        user_id = tf.squeeze(tf.fill(nr_rated, record['user_id']))
        rating = tf.cast(tf.squeeze(tf.boolean_mask(record['x'], record['mask'])), tf.dtypes.float32)
        return tf.data.Dataset.from_tensor_slices({'item_id': rated, 'rating': rating, 'user_id': user_id})


    def train(self, dataset: tf.data.Dataset, nr_records: int):
        dataset = dataset.shuffle(512,reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        for i in range(self.epochs):
            
            dataset = dataset.shuffle(512)
            step = 0
            epcoh_start = time.time()
            for data in dataset:
                users = data['user_id']
                rated = tf.cast(data['x'], tf.float32)
                mask =  tf.cast(data['mask'], tf.float32)
                
                loss_value, grads = train_dtn(self.model, users, rated, mask)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                diff = time.time() - epcoh_start
                if step % 10 == 0:
                    predictions = self.predict(rated, users)
                    train_accuracy = Evaluation.tf_calculate_accuracy(predictions, data['x'], data['mask'])
                    eval_x =  data['x_test']
                    eval_mask = data['mask_test']
                    eval_accuracy = Evaluation.tf_calculate_accuracy(predictions, eval_x, eval_mask)
                    
                    print("\rEpoch #{} Loss at step {}: {:.4f}, time: {:.3f}. Train accuracy {:.3f}, Validation accuracy {:.3f}"
                          .format(i, step, tf.reduce_mean(loss_value).numpy(), diff, train_accuracy, eval_accuracy), end='\r')
                else:
                    print("\rEpoch #{} Loss at step {}: {:.4f}, time: {:.3f}"
                          .format(i, step, tf.reduce_mean(loss_value).numpy(), diff), end='\r')
                step += 1
            print()
            self.epochs_trained += 1


    def save(self, path):
        pass

    def load(self, path):
        pass


    def predict_single_user(self, user):
        return self.model.predict([user])

    def predict(self, data: np.ndarray, user_ids: np.array) -> np.ndarray:
        user_ids = tf.convert_to_tensor(user_ids)
        return self.model.predict(user_ids).numpy()

    def get_name(self) -> str:
        return "NeuralLogicRec"

    def get_params(self) -> dict:
        items = np.arange(self.nr_items)
        param_names = ['embedding_dim', 'epochs_trained', 'batch_size', 'nr_hidden_layers', 'nr_item_samples']
        return  {param: (self.__dict__[param]) for param in param_names}
