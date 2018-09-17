import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# from DiscretizedEnvironment import DiscretizedEnvironment
from Environment import Environment

class ReinforcementIndicator:

    def __init__(self):
        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        n_hidden_2 = 256 # 2nd layer number of features
        n_input = 6 #49 # Num states
        n_classes = 2 #7 # Q-value for each action

        self.num_actions = n_classes
        self.num_states = n_input

        # tf Graph input
        self.x = tf.placeholder(tf.float32, shape=[1, n_input])

        # Store layers weight &amp; bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes])),
        }

        # Construct model
        self.neuralNetwork = self.multilayer_perceptron(
            self.x, self.weights, self.biases
        )

    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with ReLU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with ReLU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

        return out_layer

    def train(self):

        Qout = self.neuralNetwork
        predict = tf.argmax(Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        nextQ = tf.placeholder(shape=[1,self.num_actions],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        updateModel = trainer.minimize(loss)

        init = tf.global_variables_initializer()

        # Set learning parameters
        y = .99
        e = 0.1
        num_epochs = 100
        time_horizon = 1000

        env = Environment() # DiscretizedEnvironment()

        #create a list to contain total rewards
        rList = []
        with tf.Session() as sess:
            sess.run(init)
            for i in range(num_epochs):
                print "epoch: ", i
                # Reset environment and get first new observation
                s = env.reset()
                rAll = 0
                j = 0
                # The Q-Network
                while j < time_horizon:
                    j+=1
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    a,allQ = sess.run(
                        [predict,Qout],
                        feed_dict={
                            self.x:np.identity(self.num_states)[s:s+1]
                        }
                    )

                    if np.random.rand(1) < e:
                        a[0] = np.random.choice(env.action_space)
                    # Get new state and reward from environment
                    s1,r = env.step(a[0])
                    # Obtain the Q' values by feeding the new state through our network
                    Q1 = sess.run(
                        Qout,
                        feed_dict={
                            self.x:np.identity(self.num_states)[s1:s1+1]
                        }
                    )
                    # Obtain maxQ' and set our target value for chosen action.
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0,a[0]] = r + y*maxQ1
                    #Train our network using target and predicted Q values
                    _,weights1, biases1 = sess.run(
                        [updateModel, self.weights, self.biases],
                        feed_dict={
                            self.x:np.identity(self.num_states)[s:s+1],
                            nextQ:targetQ}
                    )
                    rAll += r
                    s = s1
                if rAll > 1000:
                    #Reduce chance of random action as we train the model.
                    e = 1./((i/50) + 10)

                rList.append(rAll)

        plt.plot(rList)
        plt.show()
        print np.max(rList), np.min(rList), np.mean(rList)
        print (np.asarray(rList) > 0).sum()
