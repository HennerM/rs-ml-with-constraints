from models import BaseModel
from models.ConstraintAutoRec import ConstraintAutoRec
from utils.common import movie_lens, load_dataset, load_testset
import numpy as np

def calculate_MSE(predictions, actual, with_held):
    error = np.square(predictions - actual)
    total_error = 0
    for i in range(error.shape[0]):
        with_held_items = with_held[i].nonzero()[0]
        total_error += error[i, with_held_items].mean()

    return total_error / float(error.shape[0])

def calculate_accuracy(predictions, actual, mask):
    #   tp / (tp + fp)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fp = (pred * (~actual) * mask).sum()
    return (tp + fp) / mask.sum()


def calculate_precision(predictions, actual, mask):
    #   tp / (tp + fp)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fp = (pred * (~actual) * mask).sum()
    return tp / (tp + fp)

def calculate_recall(predictions, actual, mask):
    #   tp / (tp + fn)
    pred = predictions > 0.5
    tp = (pred * actual * mask).sum()
    fn = ((~pred) * actual * mask).sum()
    return tp / (tp + fn)


def evaluate(model: BaseModel, dataset: dict):

    dataset = load_testset(dataset).batch(64)
    for batch in dataset:
        x = batch['x'].numpy()
        y = batch['y'].numpy()
        mask = batch['mask'].numpy()
        with_held = batch['held_back'].numpy()

        predictions = model.predict(x)
        print("Accuracy:", calculate_accuracy(predictions, y, mask))
        print("Precision:", calculate_precision(predictions, y, mask))
        print("Recall:", calculate_recall(predictions, y, mask))
        print("MSE", calculate_MSE(predictions, y, with_held))

    return predictions, y


if __name__ == "__main__":
    model = ConstraintAutoRec(movie_lens['dimensions'])
    model.train(load_dataset(movie_lens, 'train'), movie_lens['train']['records'])

