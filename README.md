# minimal-nn

A minimal implementation of neural network for MNIST experiment. Used as an exercise to understand Backpropagation by implementing it in NumPy.

# Dependencies

- Python 2.7
- numpy
- matplotlib

# Instructions

Neural networks have two modes: inference/prediction and learning/training. They are implemented as forward propagation and backward propagation (backpropagation for short). 

`fnn.py` contains a minimal implementation of multi-layer feedforward neural network. The main class is `FNN` that holds a list of layers, and defines the high level iterative process for forward and backward propagation. Class `Layer` implements each layer in the neural network. Class GradientDescentOptimizer implements an optimizer for training the neural network. The utility functions at the end implements different activation functions, loss functions and their gradients. Read through `fnn.py` to get an overview of the implementation. Like most effcient implementations of neural network, we are using minibatch gradient descent instead of stochastic gradient descent, see [this video](https://www.youtube.com/watch?v=qfDAtjJrquc) to learn more.

- Forward propagation: the core of inference/prediction in NN.
In this part, you need to complete the forward method in class Layer in `fnn.py` (search for `Question 1` to see the instructions). (2 lines of code)

- Backpropagation: the core of learning/training in NN.
In this part, you need to complete the backward method in class Layer in `fnn.py` (search for `Question 2` to see the instructions). (4 lines of code)

Read [this notes](http://cs231n.github.io/optimization-2/) on intuition and implementation tips for Backpropagation. `Backprop in practice: Staged computation` and `Gradients for vectorized operations` sections are especially helpful with good examples and practical tips. 

First, download the MNIST dataset by running
```
python get_mnist_data.py
```

To test your implementation, run
```
python test_fnn.py
```

There are two tests `test_forwardprop` and `test_backprop`. When your implementation passes both of them, run 
```
python mnist_experiment.py
```

to train a small deep neural network with 2 hidden layers (containing 128 and 32 RELU units each) for handwritten digits recognition using [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The accuracy should be around 99% on training set and around 97% on validation and test set. 

To demonstrate the effect of learning, 100 randomly selected test images will be shown with true labels (black on top left corner), predictions before training (red on bottom right corner), and predictions after training (blue on bottom left corner). See the figure below as an example. You can see that the predictions improve from random guess to almost perfect. Yes, it learns :) 

![Example figure](/example_images.png?raw=true "Figure 1")

