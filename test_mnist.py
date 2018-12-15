from __future__ import (absolute_import, division, print_function)
import numpy as np
import tensorflow as tf

# load local model
from cnn_mnist import cnn_model_fn


def main(unused_argv):
    """MNIST estimator"""
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    (train_data_a, train_labels_a), (eval_data_a, eval_labels_a) = tf.keras.datasets.mnist.load_data()

    # train data
    train_data = np.asarray(train_data_a, dtype=np.float32)
    train_labels = np.asarray(train_labels_a, dtype=np.int32)

    # eval data
    eval_data = np.asarray(eval_data_a, dtype=np.float32)
    eval_labels = np.asarray(eval_labels_a, dtype=np.int32)

    # train_data = mnist.train.images
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    
    # # eval data
    # eval_data = mnist.test.images
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # create an Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="cnn_mnist_model")

    # logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=2000,
        hooks=[logging_hook]
    )

    # evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


if __name__ == '__main__':
    # set logging verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    # invoke
    tf.app.run(main=main)