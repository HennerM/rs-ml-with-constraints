from models import BaseModel
from models.ConstraintAutoRec import ConstraintAutoRec
from utils.common import movie_lens, load_dataset


def evaulate(model: BaseModel, dataset: dict):
    model.train(load_dataset(dataset, 'train'), dataset['train']['records'])

    model.test()


model = ConstraintAutoRec(movie_lens['dimensions'])
