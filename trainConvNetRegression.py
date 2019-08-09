import os
from datetime import datetime, timedelta

import numpy as np
import pandas
import tensorflow as tf

from ConvNetRegression import Learner

batch_size = 1000
feature_size = 1024

csv = "HeartRate.pruned.1025.csv" # 104210 chunks * 1025 samples + 1 header line
csvTest = "HeartRate.June2019.1025.csv" # 2072 * 1025 + 1

# for running tensorboard
# release the port first
# sudo fuser 6006/tcp -k
# tensorboard --logdir=TfSummaries
# browse localhost:6006

def readFile(fileName):
    df = pandas.read_csv(fileName, usecols=["rate", "v0", "v1", "v2", "v3"])

    r = df["rate"][::feature_size+1]
    x = df[["v0", "v1", "v2", "v3"]]
    x = np.reshape(x.values, (-1, feature_size + 1, x.shape[1]))

    x, r = filterData(x, r)

    return x, np.asarray(r, dtype=np.int32)

def filterData(x, r):
    rMax = 140

    x = [row for i, row in enumerate(x) if r.values[i] <= rMax]
    x = np.asarray(x)
    r = [row for i, row in enumerate(r) if row <= rMax]
    return x, r

def evaluate(classifier, eval_data, eval_labels):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    return eval_results

def main(unused_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable verbose logging
    Run()
    # Test()

def Test():
    learner = Learner(batch_size)
    eval_data, eval_labels = readFile(csvTest)

    classifier = tf.estimator.Estimator(model_fn=learner.model, model_dir='TfSummaries/saved')
    eval_results = evaluate(classifier, eval_data, eval_labels)

    print(eval_results)

def Run():

    train_data, train_labels = readFile(csv)
    test_n = len(train_data) - len(train_data) % batch_size
    train_data = train_data[:test_n]
    train_labels = train_labels[:test_n]

    eval_data, eval_labels = readFile(csvTest)

    with tf.device('/gpu:0'):
        learner = Learner(batch_size)

        summariesDir = 'TfSummaries/%s%s' % (datetime.now().strftime('%b%d_%H:%M'), learner.schema())

        classifier = tf.estimator.Estimator(model_fn=learner.model, model_dir=summariesDir)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            queue_capacity=30000,
            shuffle=True)

        #logging_hook = tf.train.LoggingTensorHook(tensors={"debug": "model/debug"}, every_n_secs=3)

        while True:
            classifier.train(input_fn=train_input_fn, steps=100000)#, hooks=[logging_hook])

            eval_results = evaluate(classifier, eval_data, eval_labels)

            print(eval_results)

if __name__ == "__main__":
    tf.app.run()
