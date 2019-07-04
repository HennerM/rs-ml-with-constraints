import tensorflow as tf
from models.ConstraintAutoRec import ConstraintAutoRec
import datetime
import os

from utils.common import movie_lens, load_dataset

model = ConstraintAutoRec(movie_lens['dimensions'])
model.train(load_dataset(movie_lens), movie_lens['train']['records'])


# today = datetime.date.today()
# directory = 'saved_models/' + str(today) + '/'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# model.save(directory)

