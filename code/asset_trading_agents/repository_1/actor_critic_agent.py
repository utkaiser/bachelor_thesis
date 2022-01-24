import numpy as np
import tensorflow as tf
import random
import asset_trading_agents.repository_1.agent as agent

class Actor:
    def __init__(self, name, input_size, output_size, size_layer):
        with tf.compat.v1.variable_scope(name):
            self.X = tf.compat.v1.placeholder(tf.float32, (None, input_size))
            feed_actor4 = tf.compat.v1.layers.dense(self.X, size_layer, activation = tf.nn.relu)
            self.logits = tf.compat.v1.layers.dense(feed_actor4, output_size)


class Critic:
    def __init__(self, name, input_size, output_size, size_layer, learning_rate):
        with tf.compat.v1.variable_scope(name):
            self.X = tf.compat.v1.placeholder(tf.float32, (None, input_size))
            self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
            self.REWARD = tf.compat.v1.placeholder(tf.float32, (None, 1))
            feed_critic1 = tf.compat.v1.layers.dense(self.X, size_layer, activation = tf.nn.relu)
            feed_critic2 = tf.compat.v1.layers.dense(feed_critic1, output_size, activation = tf.nn.relu) + self.Y
            feed_critic3 = tf.compat.v1.layers.dense(feed_critic2, size_layer//2, activation = tf.nn.relu)
            self.logits = tf.compat.v1.layers.dense(feed_critic3, 1)
            self.cost = tf.reduce_mean(input_tensor=tf.square(self.REWARD - self.logits))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, name="2").minimize(self.cost)


class Agent(agent.Abstract_Agent):

    def __init__(self, trading_env, decay_rate,layer_size,gamma,memory_size):
        tf.compat.v1.reset_default_graph()
        super().__init__(trading_env)
        self.decay_rate = decay_rate
        self.layer_size = layer_size
        self.gamma = gamma
        self.name = "actor_critic"
        self.learning_rate_critic = 0.001
        self.learning_rate_optimizer = 0.001
        self.memory_size = memory_size
        self.actor = Actor('actor-original', self.state_size, self.output_size, self.layer_size)
        self.actor_target = Actor('actor-target', self.state_size, self.output_size, self.layer_size)
        self.critic = Critic('critic-original', self.state_size, self.output_size, self.layer_size,
                             self.learning_rate_critic)
        self.critic_target = Critic('critic-target', self.state_size, self.output_size,
                                    self.layer_size, self.learning_rate_critic)
        self.grad_critic = tf.gradients(ys=self.critic.logits, xs=self.critic.Y)
        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32, [None, self.output_size])
        weights_actor = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.grad_actor = tf.gradients(ys=self.actor.logits, xs=weights_actor, grad_ys=-self.actor_critic_grad)
        grads = zip(self.grad_actor, weights_actor)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate_optimizer).apply_gradients(grads_and_vars=grads)
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _assign(self, from_name, to_name):
        from_w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=from_name)
        to_w = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=to_name)
        for i in range(len(from_w)):
            assign_op = to_w[i].assign(from_w[i])
            self.sess.run(assign_op)

    def _memorize(self, state, action, reward, new_state, dead):
        self.memory.append((state, action, reward, new_state, dead))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def _select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.output_size)
        else:
            prediction = self.sess.run(self.actor.logits, feed_dict={self.actor.X:[state]})[0]
            action = np.argmax(prediction)
        return action

    def _construct_memories_and_train(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.sess.run(self.actor.logits, feed_dict={self.actor.X: states})
        Q_target = self.sess.run(self.actor_target.logits, feed_dict={self.actor_target.X: states})
        grads = self.sess.run(self.grad_critic, feed_dict={self.critic.X:states, self.critic.Y:Q})[0]
        self.sess.run(self.optimizer, feed_dict={self.actor.X:states, self.actor_critic_grad:grads})

        rewards = np.array([a[2] for a in replay]).reshape((-1, 1))
        rewards_target = self.sess.run(self.critic_target.logits,
                                       feed_dict={self.critic_target.X:new_states,self.critic_target.Y:Q_target})
        for i in range(len(replay)):
            if not replay[0][-1]:
                rewards[i] += self.gamma * rewards_target[i]
        cost, _ = self.sess.run([self.critic.cost, self.critic.optimizer],
                                feed_dict={self.critic.X:states, self.critic.Y:Q, self.critic.REWARD:rewards})
        return cost

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

            for t in range(0, len(df_training) - 1):
                self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * np.exp(-self.decay_rate * i)
                self.action = self._select_action(self.state)
                self.next_state = self.get_state(t + 1, df_training)

                self.take_action_training(t, df_training)

                invest = ((self.starting_money - self.initial_money) / self.initial_money)

                self._memorize(self.state, self.action, invest, self.next_state, self.starting_money < self.initial_money)
                self.state = self.next_state
                replay = random.sample(self.memory, len(self.memory))
                cost = self._construct_memories_and_train(replay)

            if self.early_stopping(self.starting_money):
                self.continue_loop = False

            if (i + 1) % self.checkpoint == 0:
                self.log_current_epoch(i + 1, cost, self.starting_money)
            i += 1

    def buy(self, df_test):

        self.reset_test_variables()
        self.state = self.get_state(t=0, trend=df_test)  # returns difference array
        t = 0

        for t in range(0, len(df_test) - 1):  # iterate over selected days, t = timestamp
            self.action = self._select_action(self.state)
            self.next_state = self.get_state(t + 1, df_test)
            self.take_action_testing(t, df_test)

        self.determine_results(df_test[t])
        return self.states_buy, self.states_sell, self.total_gains, self.invest, self.portfolio_movement












