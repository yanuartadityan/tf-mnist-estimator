## TensorFlow MNIST CNN with Estimator

This is a quick straightforward 2 convolutional layers plus max pooling feed-forward fully connected (Dense) neural network to solve MNIST classification problems. 

## Reference

I am using high level TensorFlow's [Estimator][1], a great way to simplify your ML programming. The database used is already popular [MNIST][2] database containing handwritten digits. 

## Remarks and Ongoing Works

* Only tested in [TensorFlow 1.10.0][3]
* Only has been trained with ~2000 steps
* (Ongoing) Adding detailed visualization on each layer in TensorBoard


[1]: https://www.tensorflow.org/guide/custom_estimators
[2]: http://yann.lecun.com/exdb/mnist/
[3]: https://github.com/tensorflow/tensorflow/releases
