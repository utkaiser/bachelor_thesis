import random
import numpy as np
import tensorflow as tf
import asset_trading_agents.repository_1.agent as agent

class Agent(agent.Abstract_Agent):

    def __init__(self, trading_env, decay_rate,layer_size,gamma, memory_size):
        tf.compat.v1.reset_default_graph()
        super().__init__(trading_env)
        self.name = "q_learning_agent"
        self.decay_rate = decay_rate
        self.layer_size = layer_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.learning_rate = 0.01
        self.sess = tf.compat.v1.InteractiveSession()
        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.state_size])
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.action_size])
        feed = tf.compat.v1.layers.dense(self.X, self.layer_size, 'relu')
        self.logits = tf.compat.v1.layers.dense(feed, self.action_size)
        self.cost = tf.reduce_mean(input_tensor=tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(
            self.cost
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())


    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        prediction = self.sess.run(self.logits, feed_dict = {self.X: state})[0]
        return np.argmax(prediction)

    def get_state(self, t, trend):
        d = t - self.state_size
        block = trend[d : t + 1] if d >= 0 else -d * [trend[0]] + trend[0 : t + 1]
        res = []
        for i in range(self.state_size):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def replay(self, replay, batch_size):
        mini_batch = []
        l = len(replay)
        for i in range(l - batch_size, l):
            mini_batch.append(replay[i])
        replay_size = len(mini_batch)
        X = np.empty((replay_size, self.state_size))
        Y = np.empty((replay_size, self.action_size))
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            print("target",len(target),target)
            target[action] = reward
            if not done:
                target[action] += self.gamma * np.amax(Q_new[i])
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
                self.action = self.act(self.state)
                self.next_state = self.get_state(t + 1, df_training)
                self.take_action_training(t, df_training)
                invest = ((self.starting_money - self.initial_money) / self.initial_money)
                self.memory.append((self.state, self.action, invest,
                                    self.next_state, self.starting_money < self.initial_money))
                if len(self.memory) > self.memory_size:
                    self.memory.popleft()
                self.state = self.next_state
                replay = random.sample(self.memory, len(self.memory))
                cost = self.replay(replay,len(self.memory))

            if self.early_stopping(self.starting_money):
                self.continue_loop = False

            if (i + 1) % self.checkpoint == 0:
                self.log_current_epoch(i + 1, cost, self.starting_money)
            i += 1

    def buy(self, df_test):

        self.reset_test_variables()
        self.state = self.get_state(t=0, trend=df_test)  # returns difference array
        t = 0

        for t in range(0, len(df_test) - 1):
            self.action = self.act(self.state)
            self.next_state = self.get_state(t + 1,trend=df_test)
            self.take_action_testing(t,df_test)

        self.determine_results(df_test[t])
        return self.states_buy, self.states_sell, self.total_gains, self.invest, self.portfolio_movement









