# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import cifar10
import csv
from datetime import datetime

FLAGS = None

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
batch_sizes = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
range_array = [500, 1000, 5000, 10000]

def img_to_r(img):
    result = []
    for x in range(32):
        for y in range(32):
          for c in range(3):
            result.append(img[x][y][c])
    return result

images_train, cls_train, labels_train = cifar10.load_training_data()
images_train_prepared = list(map(img_to_r, images_train))

images_test, cls_test, labels_test = cifar10.load_test_data()
images_test_prepared = list(map(img_to_r, images_test))

def take_batch(list, offset, size):
    index = offset * size
    return list[index:index+size]

def save_in_file(file_name, data):
  with open(file_name,'a') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerows([[data]][0])


def main(_):
  cifar10.maybe_download_and_extract()

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  img_size_x = 32
  img_size_y = 32
  img_vec = img_size_x * img_size_y * 3 

  # Create the model
  x = tf.placeholder(tf.float32, [None, img_vec])
  W = tf.Variable(tf.zeros([img_vec, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  for single_range in range_array:
    for learning_rate in learning_rates:
      for batch_size in batch_sizes: 
        try:     
          cross_entropy = tf.reduce_mean(
              tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
          train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

          sess = tf.InteractiveSession()
          tf.global_variables_initializer().run()


          # Train
          for i in range(single_range):
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = take_batch(images_train_prepared, i, batch_size)
            batch_ys = take_batch(labels_train, i, batch_size)
            # [[0..783]...[]]
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

          # Test trained model
          correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
          #print(sess.run(accuracy, feed_dict={x: images_test_prepared, y_: labels_test}))

          save_in_file("results.csv", [str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), str(single_range),str(learning_rate), str(batch_size), str(sess.run(accuracy, feed_dict={x: images_test_prepared, y_: labels_test}))])
        except Exception as e:
          save_in_file("results.csv", [str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), str(single_range), str(learning_rate), str(batch_size), str(sess.run(accuracy, feed_dict={x: images_test_prepared, y_: labels_test})), str(e)])
        sess.close()
  # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
  #                                     y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
