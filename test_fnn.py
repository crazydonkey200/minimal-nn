import json
import unittest
import numpy as np

import fnn
import mnist_experiment as me
import check_grad as cg


# A tiny subset from the mnist dataset.
with open('toy_dataset.json', 'r') as f:
    toy_dataset = json.load(f)

toy_data = np.array(toy_dataset['data'][:10])
toy_labels = np.array(toy_dataset['labels'][:10])

# Data and labels to test the simple example we went through
# in the class.
data = np.array([[1, 2], [3, 1]])
labels = np.array([[0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 0]])

class TestFNN(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=349)
        # Setup model to test on the tiny subset of mnist.
        self.m1 = fnn.FNN(784, 10, [16, 8], [fnn.relu, fnn.relu])
        # Used to check the whether bias is dealt correctly
        # in forwardprop.
        self.m1.layers[-1].b[0][0] = 1.0

        # Setup another model to test on the simple example
        # we went through in class
        self.m2 = fnn.FNN(2, 5, [3, 2], [fnn.relu, fnn.relu])
        self.m2.layers[0].w = np.array([[1, -3, 4],
                                        [-2, 1, 2]])
        self.m2.layers[0].b = np.array([[2, -1, 1]])
        self.m2.layers[1].w = np.array([[1, -1],
                                        [2, 1],
                                        [-1, 2]])
        self.m2.layers[1].b = np.array([[1, -1]])
        self.m2.layers[2].w = np.array([[1, -2, 2, -2, 1],
                                        [-1, 1, 1, -1, -1]])
        self.m2.layers[2].b = np.array([[1, 0, -1, 1, 2]])
        
    def test_forwardprop(self):
        # Test on the tiny subset of mnist.
        probs, loss = self.m1.forwardprop(toy_data, toy_labels)
        self.assertTrue(abs(loss - 2.3677889) < 0.0000001)

        # Test on the simple example.
        self.m2.forwardprop(data)
        # Check the activations of the first two layers.
        self.assertTrue(np.allclose(self.m2.layers[0].a,
                                    np.array([[0, 0, 9],
                                              [3, 0, 15]])))

        self.assertTrue(np.allclose(self.m2.layers[1].a,
                                    np.array([[0, 17],
                                              [0, 26]])))
        print '\n' + '=' * 50 + '\n'
        print "Your forward propagation is correct!"
        print '\n' + '=' * 50 + '\n'

    def test_backprop(self):
        # Use gradient check to test on the tiny mnist
        # subset.
        self.assertTrue(
            cg.check_backprop(self.m1, toy_data, toy_labels) < 1e-4)

        # Use pre-computed gradients to test on the simple
        # example.
        self.m2.forwardprop(data)
        self.m2.backprop(labels)
        # Check the gradients of the loss w.r.t the first
        # two layers' weights and bias.
        self.assertTrue(np.allclose(self.m2.layers[1].d_w,
                                    np.array([[0, 3],
                                              [0, 0],
                                              [0, 15]])))

        self.assertTrue(np.allclose(self.m2.layers[1].d_b,
                                    np.array([[0, 1]])))

        self.assertTrue(np.allclose(self.m2.layers[0].d_w,
                                    np.array([[-3, 0, 6],
                                              [-1, 0, 2]])))

        self.assertTrue(np.allclose(self.m2.layers[0].d_b,
                                    np.array([[-1, 0, 2]])))
        print '\n' + '=' * 50 + '\n'
        print "Your backpropagation is correct!"
        print '\n' + '=' * 50 + '\n'


if __name__ == '__main__':
    unittest.main()
