import tensorflow as tf
import numpy as np


class FFNN(object):
    def __init__(self, layers, activation=tf.tanh, output_activation=tf.identity, post_function=tf.identity, input_vector=None, session=None):
        """ An implementation of a simple feed-forward neural network using the low-level
            tensorflow API.

        Inputs
            layers <list(ints)>: A list of integers giving the number of nodes in each layer. The
                first item in the list gives the input dimension and the last item gives output
                dimension.
            activation <function(tf.Tensor -> tf.Tensor)>: A function mapping tensors to tensors
                which will be used as the activation function.
            output_activation <function(tf.Tensor -> tf.Tensor)>: Neural networks often apply
                a softmax or different form of computation to the final output layer. This argument
                specifies the function that will be used. Defaults to the identity map.
            input_vector <tf.Tensor>: An (None, dim_in) shaped tensor used as the input layer to
                the network. This can be used to link the network to more complex structures. If
                not given, a tensorflow placeholder is initialized and used instead.
            session <tf.Session>: A tensorflow session object. This argument is provided as an
                option so that to prevent graph ownership issuses when an FFNN is used as a
                component of a larger network. If not provided, a new interactive session will be
                initialized.

        Attributes
            session
            activation
            layers
            input
            output
            weights
            biases
        """
        # Set basic attributes
        self.activation = activation
        self.output_activation = output_activation
        self.post_function = post_function
        self.layers = layers
        self.learning_curve = []
        self.epochs = 0

        if session is None:
            self.session = tf.InteractiveSession()
        else:
            self.session = session

        if input_vector is None:
            self.input = tf.placeholder(tf.float32, [None, layers[0]])
        else:
            self.input = input_vector

        self.output = self.input
        self.train_targets = tf.placeholder(tf.float32, [None, layers[-1]])

        # Initialize Weights
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            self.weights.append(tf.Variable(tf.random_uniform([self.layers[i], self.layers[i + 1]], -1, 1)))
            self.biases.append(tf.Variable(tf.random_uniform([self.layers[i + 1]], -1, 1)))

        # Construct the model calculation in the graph
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            self.output = self.activation(tf.matmul(self.output, w) + b)
        # The last layer has its own activation applied
        self.output = self.output_activation(tf.matmul(self.output, self.weights[-1]) + self.biases[-1])

        # Initialize variables
        self.session.run(tf.global_variables_initializer())


    def train(self, train_in, train_out, loss_func=None, optimizer=None, batch_size=10, epochs=1, lc_interval=10):
        """ Train the network weights with provided data. Trained weights can be accessed inside of
            the tensorflow session stored as `self.session`.

        Inputs
            train_in <np.ndarray>: A [n, d] shaped array where n is the number of datapoints and d
                is size of the input dimension.
            train_out <np.ndarray>: A [n, c] shaped array where n is the number of datapoints and c
                is size of the output dimension.
            loss_func
            optimizer
            batch_size
            epochs
        """
        if loss_func is None: loss_func = tf.losses.mean_squared_error
        if optimizer is None: optimizer = tf.train.GradientDescentOptimizer(0.01)

        loss_val = loss_func(self.output, self.train_targets)
        train_step = optimizer.minimize(loss_val)

        for i in range(epochs):
            in_batch = np.roll(train_in, -batch_size * i, 0)[:batch_size]
            out_batch = np.roll(train_out, -batch_size * i, 0)[:batch_size]
            self.session.run(train_step, feed_dict={self.input: in_batch, self.train_targets: out_batch})

            if self.epochs % lc_interval == 0:
                self.learning_curve.append((
                    self.epochs,
                    self.session.run(loss_val, feed_dict={self.input: train_in, self.train_targets: train_out})
                ))
                print("Reached epoch {}".format(self.epochs))
            self.epochs += 1


    def evaluate(self, in_vector):
        """ Runs the model on a numpy array representing a collection of input data

        Inputs
            in_vector
        Returns
            <np.ndarray>
        """
        return self.session.run(self.post_function(self.output), feed_dict={self.input: in_vector})

