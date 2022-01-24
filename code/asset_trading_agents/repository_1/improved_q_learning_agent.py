import random
import numpy as np
import tensorflow as tf
import asset_trading_agents.repository_1.agent_improved as agent

class Agent(agent.Abstract_Agent):
    def __init__(self, trading_env, decay_rate,layer_size,gamma,memory_size,dropout,learning_rate):
        tf.compat.v1.reset_default_graph()
        super().__init__(trading_env)
        self.name = "q_learning_agent"
        self.decay_rate = decay_rate
        self.layer_size = layer_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.max_action = 10
        self.sess = tf.compat.v1.InteractiveSession()
        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.state_size*6])
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.action_size])
        feed = tf.compat.v1.layers.dense(self.X, self.layer_size, 'relu')
        tf_is_traing_pl = tf.compat.v1.placeholder_with_default(True, shape=())
        drop_out = tf.compat.v1.layers.dropout(feed, rate=dropout,training=tf_is_traing_pl)
        self.logits = tf.compat.v1.layers.dense(drop_out, self.action_size,'softmax')
        self.cost = tf.reduce_mean(input_tensor=tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(
            self.cost
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(-self.max_action-1,self.max_action+1), random.randrange(0,3) #no total investments, risk aversion
        prediction = self.sess.run(self.logits, feed_dict = {self.X: state})[0]*self.max_action
        index = np.argmax(prediction)
        prediction = int(prediction[index])
        if index == 2:
            prediction = -prediction
        return prediction, index

    def get_state(self, t, trend):
        d = t - self.state_size
        block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
        res = []
        for i in range(self.state_size):
            for j in range(len(block[0])):
                res.append(block[i + 1][j] - block[i][j])
        return np.array([res])

    def replay(self, replay, batch_size, index):
        mini_batch = []
        l = len(replay)
        for i in range(l - batch_size, l):
            mini_batch.append(replay[i])
        replay_size = len(mini_batch)
        X = np.empty((replay_size, self.state_size*6))
        Y = np.empty((replay_size, self.action_size))
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            target[index] = reward
            if not done:
                target[index] += self.gamma * np.amax(Q_new[i])
            X[i] = state
            Y[i] = target
        cost, _ = self.sess.run(
            [self.cost, self.optimizer], feed_dict = {self.X: X, self.Y: Y}
        )
        return cost

    def train(self, df_training):
        i = 0
        while i < self.iterations and self.continue_loop:
            self.reset_training_variables()
            self.state = self.get_state(t=0, trend=df_training)

            for t in range(0, len(df_training) - 1):
                self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * i)
                self.action, index = self.act(self.state)
                self.next_state = self.get_state(t + 1, df_training)
                self.take_action_training(t, df_training)
                invest = ((self.starting_money - self.initial_money) / self.initial_money)
                self.memory.append((self.state, self.action, invest,
                                    self.next_state, self.starting_money < self.initial_money))
                if len(self.memory) > self.memory_size:
                    self.memory.popleft()
                self.state = self.next_state
                replay = random.sample(self.memory, len(self.memory))
                cost = self.replay(replay,len(self.memory),index)

            if self.early_stopping(self.starting_money):
                self.continue_loop = False

            if (i + 1) % self.checkpoint == 0:
                self.log_current_epoch(i + 1, cost, self.starting_money)
            i += 1

    def buy(self, df_test):
        self.reset_test_variables()
        #self.sess.run({'tf_is_training': tf.compat.v1.placeholder_with_default(True, shape=())})
        self.state = self.get_state(t=0, trend=df_test)  # returns difference array
        t = 0
        for t in range(0, len(df_test) - 1):
            self.action, _ = self.act(self.state)
            self.next_state = self.get_state(t + 1,trend=df_test)
            self.take_action_testing(t,df_test)

        self.determine_results(df_test[t][0])
        return self.states_buy, self.states_sell, self.total_gains, self.invest, self.portfolio_movement









