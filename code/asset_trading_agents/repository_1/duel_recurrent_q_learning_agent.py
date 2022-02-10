import numpy as np
import tensorflow as tf
import random
import asset_trading_agents.repository_1.agent as agent
from collections import deque

class Agent(agent.Abstract_Agent):
    def __init__(self, trading_env, decay_rate,layer_size,gamma,memory_size):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        super().__init__(trading_env)
        self.name = "duel_recurrent_q_learning"
        self.decay_rate = decay_rate
        self.layer_size = layer_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.learning_rate = 0.001
        self.INITIAL_FEATURES = np.zeros((4, self.state_size))
        self.X = tf.compat.v1.placeholder(tf.float32, (None, None, self.state_size))
        self.Y = tf.compat.v1.placeholder(tf.float32, (None, self.output_size))
        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.layer_size, state_is_tuple = False)
        self.hidden_layer = tf.compat.v1.placeholder(tf.float32, (None, 2 * self.layer_size))
        self.rnn,self.last_state = tf.compat.v1.nn.dynamic_rnn(inputs=self.X,cell=cell,
                                                    dtype=tf.float32,
                                                    initial_state=self.hidden_layer)
        tensor_action, tensor_validation = tf.split(self.rnn[:,-1],2,1)
        feed_action = tf.compat.v1.layers.dense(tensor_action, self.output_size)
        feed_validation = tf.compat.v1.layers.dense(tensor_validation, 1)
        self.logits = feed_validation + tf.subtract(feed_action,tf.reduce_mean(feed_action,axis=1,keepdims=True))
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _memorize(self, state, action, reward, new_state, dead, rnn_state):
        self.memory.append((state, action, reward, new_state, dead, rnn_state))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()


    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = self.sess.run(self.logits, feed_dict={self.X:states, self.hidden_layer:init_values})
        Q_new = self.sess.run(self.logits, feed_dict={self.X:new_states, self.hidden_layer:init_values})
        replay_size = len(replay)
        X = np.empty((replay_size, 4, self.state_size))
        Y = np.empty((replay_size, self.output_size))
        INIT_VAL = np.empty((replay_size, 2 * self.layer_size))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, dead_r, rnn_memory = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not dead_r:
                target[action_r] += self.gamma * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
            INIT_VAL[i] = rnn_memory
        return X, Y, INIT_VAL

    def get_state(self, t, trend):
        d = t - self.state_size
        block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
        res = []
        for i in range(self.state_size):
            res.append(block[i + 1] - block[i])
        return np.array(res)


    def train(self, df_training):
        i = 0
        while i < self.iterations and self.continue_loop:
            self.reset_training_variables()
            self.state = self.get_state(t=0, trend=df_training)
            init_value = np.zeros((1, 2 * self.layer_size))
            for k in range(self.INITIAL_FEATURES.shape[0]):
                self.INITIAL_FEATURES[k,:] = self.state
            for t in range(0, len(df_training) - 1):
                self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * i)
                if np.random.rand() < self.epsilon:
                    self.action = np.random.randint(self.output_size)
                else:
                    self.action, last_state = self.sess.run([self.logits,
                                                  self.last_state],
                                                  feed_dict={self.X:[self.INITIAL_FEATURES],
                                                             self.hidden_layer:init_value})
                    self.action, init_value = np.argmax(self.action[0]), last_state
                self.next_state = self.get_state(t + 1,df_training)
                self.take_action_training(t, df_training)
                invest = ((self.starting_money - self.initial_money) / self.initial_money)
                new_state = np.append([self.get_state(t + 1, df_training)], self.INITIAL_FEATURES[:3, :], axis = 0)
                self._memorize(self.INITIAL_FEATURES, self.action, invest, new_state,
                               self.starting_money < self.initial_money, init_value[0])
                self.INITIAL_FEATURES = new_state
                replay = random.sample(self.memory, len(self.memory))
                X, Y, INIT_VAL = self._construct_memories(replay)
                cost, _ = self.sess.run([self.cost, self.optimizer],
                                        feed_dict={self.X: X, self.Y:Y,
                                                  self.hidden_layer: INIT_VAL})

            if self.early_stopping(self.starting_money):
                self.continue_loop = False

            if (i + 1) % self.checkpoint == 0:
                self.log_current_epoch(i + 1, cost, self.starting_money)
            i += 1

    def buy(self, df_test):
        self.reset_test_variables()
        self.state = self.get_state(t=0, trend=df_test)  # returns difference array
        t = 0
        init_value = np.zeros((1, 2 * self.layer_size))
        for k in range(self.INITIAL_FEATURES.shape[0]):
            self.INITIAL_FEATURES[k,:] = self.state
        for t in range(0, len(df_test) - 1):
            self.action, last_state = self.sess.run([self.logits,self.last_state],
                                                feed_dict={self.X:[self.INITIAL_FEATURES],
                                                            self.hidden_layer:init_value})
            self.action, init_value = np.argmax(self.action[0]), last_state
            self.next_state = self.get_state(t + 1, df_test)
            self.take_action_testing(t,df_test)
            new_state = np.append([self.get_state(t + 1, df_test)], self.INITIAL_FEATURES[:3, :], axis = 0)
            self.INITIAL_FEATURES = new_state
        self.determine_results(df_test[t])
        return self.states_buy, self.states_sell, self.total_gains, self.invest, self.portfolio_movement





