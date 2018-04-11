# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import sys
import pandas as pd
from matplotlib import pyplot
from DataPipeline import DataPipeline


EPOCHS = 100
BATCH_SIZE = 50
EVAL_BATCH_SIZE = 15
STEPS = (573 / BATCH_SIZE) * EPOCHS
EVAL_STEPS = (156 / EVAL_BATCH_SIZE) * EPOCHS
LEARNING_RATE = 0.01

pipeline = DataPipeline('./csv/train/combined.csv', './csv/test/combined.csv')


def init_model():
    
    model = tf.estimator.DNNClassifier(
        hidden_units=[9, 9],
        n_classes=3,
        model_dir='./models',
        feature_columns=pipeline.get_feature_columns(),
        activation_fn=tf.nn.relu,
        optimizer=tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-6)
    )

    # model = tf.estimator.DNNRegressor(
    #     hidden_units=[6, 3], 
    #     feature_columns=pipeline.get_feature_columns(), 
    #     model_dir='./models', 
    #     activation_fn=tf.nn.relu)

    return model

def train_model(model):
    for epoch in range(EPOCHS + 1):
        print("Starting Epoch {}/{}".format(epoch, EPOCHS))

        model.train(input_fn=lambda:pipeline.training_input_fn(batch_size=BATCH_SIZE), steps=STEPS)
        eval_dict = model.evaluate(input_fn=lambda:pipeline.evaluate_input_fn(batch_size=BATCH_SIZE), steps=STEPS)
        print(eval_dict)

        evaluate_model(model)


def print_prediction(model, date, close, volume, sma_20, upper_band, lower_band, previous_price, expectation):
    correct = False
    prediction = model.predict(lambda:pipeline.predict_input_fn(date, close, volume, sma_20, upper_band, lower_band, previous_price))
    for i, p in enumerate(prediction):
        expect_class = _price_to_class(close, expectation)
        expect_string = pipeline.get_class_as_string(expect_class)
        class_id = p['class_ids'][0]
        probabilities = p['probabilities'][class_id]
        predicted_action = pipeline.get_class_as_string(class_id)
        correct = class_id == expect_class
        print("Date: {} Prediction is {} ({}%), Expected {} {}".format(date, predicted_action, probabilities * 100, expect_string, '✔️' if correct else '❌'))

    return correct


def _price_to_class(close_price, future_price, threshold=0.025):
    percent = (future_price - close_price) / close_price
    if percent >= threshold:
        return 2
    elif percent <= (threshold * -1):
        return 0
    else:
        return 1


def evaluate_model(model):
    eval_result = model.evaluate(input_fn=lambda:pipeline.evaluate_input_fn(batch_size=EVAL_BATCH_SIZE),steps=EVAL_STEPS)
    print(eval_result)
    
    # correct_count = 0

    # correct_count = correct_count + 1 if print_prediction(model, '03-27-2018', 10.0, 86551000, 11.4315, 12.5552, 10.3078, 11.11, 9.77) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-26-2018', 10.44, 75879000, 11.5580, 12.5730, 10.5430, 11.43, 9.55) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-23-2018', 10.63, 54844000, 11.6570, 12.6002, 10.7138, 11.47, 9.53) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-22-2018', 10.91, 59944000, 11.7290, 12.5609, 10.8971, 11.46, 10.05) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-21-2018', 11.26, 44692000, 11.7755, 12.5183, 11.0327, 11.36, 9.81) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-20-2018', 11.11, 65117000, 11.7985, 12.5036, 11.0934, 11.64, 10.0) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-19-2018', 11.43, 53309000, 11.8440, 12.4795, 11.2085, 11.52, 10.44) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-16-2018', 11.47, 37591000, 11.8635, 12.4703, 11.2567, 11.70, 10.63) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-15-2018', 11.46, 66374000, 11.8995, 12.4939, 11.3051, 11.97, 10.91) else correct_count
    # correct_count = correct_count + 1 if print_prediction(model, '03-14-2018', 11.36, 80541000, 11.9365, 12.5086, 10.3644, 12.24, 11.26) else correct_count

    # print('Accuracy: {}%'.format(100 * correct_count/10.0))



def main(argv):

    model = init_model()

    if len(sys.argv) == 2:
        if sys.argv[1] == 'train':
            train_model(model)
        elif sys.argv[1] == 'evaluate':
            evaluate_model(model)
    else:
        print('Need argument: train or evaluate')
    

    # print(curve)
    # observed_lines = pyplot.plot(
    #     observed_times, observations, label="Observed", color="k")
    # predicted_lines = pyplot.plot(curve, label="Predicted", color="b")
    # pyplot.legend(handles=[observed_lines[0], predicted_lines[0]],
    #               loc="upper left")
    # pyplot.show()

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    pd.set_option('display.precision', 2)
    tf.logging.set_verbosity(tf.logging.FATAL)
    tf.app.run(main=main)