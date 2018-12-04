import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

from DiscretizedEnvironment import DiscretizedEnvironment
#from Environment import Environment

class ReinforcementIndicator:

    def __init__(
        self,
        stock
    ):
        self.dataFrame = pd.read_csv(
            'spx_intraday.csv', index_col=0, header=[0, 1]
        ).sort_index(axis=1)[stock]

        # Env Parameters
        n_input = 49 # Num states
        n_classes = 7 # Q-value for each action

        self.num_actions = n_classes
        self.num_states = n_input
        self.stock = stock

    def multilayer_perceptron(
        self,
        shape=[49,256,256,7]
    ):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, shape=[1, shape[0]])

        # Store layers weight &amp; bias
        self.weights = {}
        self.biases = {}
        for i in range(1,len(shape)):
            if i < len(shape)-1:
                self.weights['h'+str(i)] = tf.Variable(tf.random_normal([shape[i-1], shape[i]]))
                self.biases['b'+str(i)] = tf.Variable(tf.random_normal([shape[i]]))
            else:
                self.weights['out'] = tf.Variable(tf.random_normal([shape[i-1], shape[i]]))
                self.biases['out'] = tf.Variable(tf.random_normal([shape[i]]))
        # Hidden layer with ReLU activation
        layers = [self.x]
        for i in range(1,len(shape)-1):
            layer = tf.add(tf.matmul(layers[-1], self.weights['h'+str(i)]), self.biases['b'+str(i)])
            layer = tf.nn.relu(layer)
            layers.append(layer)
        # Output layer with sigmoid activation
        out_layer = tf.add(tf.matmul(layers[-1], self.weights['out']), self.biases['out'])

        return out_layer

    def train(
        self,
        # Set learning hyper parameters
        network_shape=[49,256,256,7],
        learning_rate=0.01,
        y=.99,
        e=.01,
        initialization_period=10,
        initial_offset=2000,
        num_epochs=100,
        time_horizon=1000,
        verbose=False,
        test_number=0,
    ):
        #with tf.device("/gpu:0"):
        self.network = self.multilayer_perceptron(shape=network_shape)
        Qout = self.network
        predict = tf.argmax(Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        nextQ = tf.placeholder(shape=[1,self.num_actions],dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        updateModel = trainer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.env = DiscretizedEnvironment(
            self.dataFrame,
            initializationPeriod=initialization_period,
            initialOffset=initial_offset,
        )

        #create a list to contain total rewards
        rList = []
        with tf.Session() as sess:
            sess.run(init)
            for i in range(num_epochs):
                #print "epoch: ", i
                # Reset environment and get first new observation
                s = self.env.reset()
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
                        a[0] = np.random.choice(self.env.action_space)
                    # Get new state and reward from environment
                    s1,r = self.env.step(a[0])
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
            save_path = saver.save(sess, "/tmp/reinforcement_model.ckpt")

        if(verbose):
            plt.plot(rList)
            fig_name = './'+ self.stock + '/' + self.stock + '_trainNN_' + str(test_number) + '_2.png'
            plt.savefig(fig_name)
            plt.close()
            #plt.show()
            print np.max(rList), np.min(rList), np.mean(rList)
            print (np.asarray(rList) > 0).sum()

        return np.mean(rList)

    def dev(
        self,
        shape=[49,256,256,7],
        random_iterations=50,
        dev_horizon = 500,
        initial_offset=2000,
    ):
        paramSet = []
        for i in range(random_iterations):
            #paratemers to train
            hidden_units = pow(2,int(round(2*np.random.rand()+7)))
            hidden_layers = int(round(2*np.random.rand()+2))
            network_shape = [49]
            for i in range(hidden_layers):
                network_shape.append(hidden_units)
            network_shape.append(7)

            network_shape = [49,256,256,7]

            learning_rate = pow(10,-5*np.random.rand()-2)
            e = pow(10,-2*np.random.rand()-1)
            y = 1 - pow(10,-2*np.random.rand()-1)
            initialization_period = int(round(25*np.random.rand()+5))

            paramSet.append({
                'network_shape': network_shape,
                'learning_rate': learning_rate,
                'e': e,
                'y': y,
                'initialization_period': initialization_period,
                'initial_offset': initial_offset,
            })

        results = []
        for params in paramSet:
            #print 'training for params: ' + str(params)
            self.train(**params)
            results.append(self.test(dev_horizon))

        return paramSet[results.index(max(results))]

    def test(
        self,
        test_time_horizon,
        ignore_offset=0,
        verbose=False,
        test_number=0,
    ):
        Qout = self.network
        predict = tf.argmax(Qout,1)
        s = self.env.getCurrentState()

        #create a list to contain total rewards
        rList = []
        rAll = 0
        rAllList = []

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "/tmp/reinforcement_model.ckpt")

            for i in range(test_time_horizon):
                # Choose an action by from the Q-network
                a,allQ = sess.run(
                    [predict,Qout],
                    feed_dict={
                        self.x:np.identity(self.num_states)[s:s+1]
                    }
                )
                # Get new state and reward from environment
                s,r = self.env.step(a[0])
                if(i >= ignore_offset):
                    rAll += r
                    rAllList.append(rAll)
                    rList.append(r)

        if(verbose):
            plt.plot(rAllList)
            fig_name = './'+ self.stock + '/' + self.stock + '_testNN_' + str(test_number) + '_2.png'
            plt.savefig(fig_name)
            plt.close()
            #plt.show()
            print np.max(rList), np.min(rList), np.mean(rList), np.max(rAllList)
            print (np.asarray(rList) > 0).sum()

        return np.sum(rList)

    def testSelected(
        self,
    ):
        if not os.path.exists(self.stock):
            os.makedirs(self.stock)

        orig_stdout = sys.stdout
        file_name = './'+ self.stock + '/' + self.stock + '_NN_2.txt'
        f = open(file_name, 'w')
        sys.stdout = f

        allRes = []
        for i in range(5):
            params = self.dev(dev_horizon=500, initial_offset=i*1000)
            print 'training with: ' + str(params)
            params['initial_offset'] = i*1000
            params['verbose'] = True
            params['test_number'] = i
            self.train(**params )
            self.test(1000,ignore_offset=500, verbose=True, test_number=i)
        #plt.plot(allRes)
        #plt.show()
        print (np.asarray(allRes) > 0).sum()

        sys.stdout = orig_stdout
        f.close()
