# Author: Mattia Silvestri

import tensorflow as tf


class FullyConnectedModel:
    def __init__(self, num_states, num_actions, batch_size, scope, apply_softmax=False, duel=False):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        self.scope = scope
        # True if you want to apply softmax
        self.apply_softmax = apply_softmax
        # now setup the model
        self._define_model(scope, duel)

    def _define_model(self, scope, duel):
        print("Building Tensorflow graph with name {}...".format(scope))
        with tf.variable_scope(self.scope):
            self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
            self._y_train = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

            # create a couple of fully connected hidden layers
            fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(),
                                  name="fc1")
            fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(),
                                  name="fc2")
            if not duel:
                self._logits = tf.layers.dense(fc2, self._num_actions, name="logits")
            else:
                self.adv = tf.layers.dense(fc2, self._num_actions, name="adv")
                self.val = tf.layers.dense(fc2, 1, name="val")
                self._logits = self.val + tf.subtract(self.adv,
                                                        tf.reduce_mean(self.adv, axis=1, keepdims=True))

            '''if self.apply_softmax:
                self._probs = tf.nn.softmax(self._logits)'''

            self.loss = tf.reduce_mean(tf.square(self._y_train - self._logits), name="loss")
            self._optimizer = tf.train.AdamOptimizer(name="optimizer").minimize(self.loss)

            self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        loss, _ = sess.run([self.loss, self._optimizer],
                           feed_dict={self._states: x_batch,
                                             self._y_train: y_batch})

        return loss

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


class CNNModel:
    def __init__(self, num_states, num_actions, batch_size, scope, duel):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        self.scope = scope
        # now setup the model
        self._define_model(scope, duel)

    def _define_model(self, scope, duel):
        print("Building Tensorflow graph with name {}...".format(scope))
        with tf.variable_scope(scope):
            self._states = tf.placeholder(shape=[None, *self._num_states], dtype=tf.float32)
            self._y_train = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)

            # create convolutional layers
            conv1 = tf.layers.conv2d(self._states,
                                     filters=32,
                                     kernel_size=[8, 8],
                                     strides=[4, 4],
                                     activation=tf.nn.relu,
                                     name="conv1")

            conv2 = tf.layers.conv2d(conv1,
                                     filters=64,
                                     kernel_size=[4, 4],
                                     strides=[2, 2],
                                     activation=tf.nn.relu,
                                     name="conv2")

            conv3 = tf.layers.conv2d(conv2,
                                     filters=64,
                                     kernel_size=[3, 3],
                                     strides=1,
                                     activation=tf.nn.relu,
                                     name="conv3")

            flat = tf.layers.flatten(conv3, name="flatten")

            fc = tf.layers.dense(flat, units=512, activation=tf.nn.relu, name="fc")

            if not duel:
                self._logits = tf.layers.dense(fc, self._num_actions, name="logits")
            else:
                self.adv = tf.layers.dense(fc, self._num_actions, name="adv")
                self.val = tf.layers.dense(fc, 1, name="val")
                self._logits = self.val + tf.subtract(self.adv,
                                                        tf.reduce_mean(self.adv, axis=1, keepdims=True))


            self.loss = tf.reduce_mean(tf.square(self._y_train - self._logits), name="loss")
            self._optimizer = tf.train.AdamOptimizer(learning_rate=.00025, name="optimizer").minimize(self.loss)

            self._var_init = tf.global_variables_initializer()

    # predict Q-values given a state
    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, *self.num_states)})

    # predict Q-values given a batch of states
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    # train the model given the input state and expected next state Q-values
    def train_batch(self, sess, x_batch, y_batch):
        loss, _ = sess.run([self.loss, self._optimizer], feed_dict={self._states: x_batch,
                                             self._y_train: y_batch})

        return loss

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init
