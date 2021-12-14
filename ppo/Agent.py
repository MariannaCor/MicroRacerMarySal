# IMPORTSsss
import math
import numpy as np
import scipy
import scipy.signal

import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


class Memory:
    def __init__(self, size, state_dim, num_action):
        # shape = (col,righe) or (righe,colonne), size deve essere il numero di colonne ...  ?
        self.states = np.zeros(shape=(size, state_dim), dtype=np.float32)
        self.actions = np.zeros(shape=(size, num_action), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)  # quella che ritorna l'environment
        self.values = np.zeros(size, dtype=np.float32)  # quella che ritorna il critico
        self.dones = np.zeros(size, dtype=np.float32)  # se l'episodio è finito

        self.advantages = np.zeros(size, dtype=np.float32)  # quella che mi serve per calcolare l'actor_loss
        self.returns = np.zeros(size, dtype=np.float32)  # quella che mi serve per calcolare il critic loss

        self.pointer, self.trajectory_start_index = 0, 0

    def store_memory(self, state, action, log_prob, reward, value, done):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.log_probs[self.pointer] = log_prob
        self.rewards[self.pointer] = reward
        self.values[self.pointer] = value
        self.dones[self.pointer] = done
        self.pointer += 1

    def get(self):
        # Get all data of the buffer
        self.pointer, self.trajectory_start_index = 0, 0

        return (
            self.states,
            self.actions,
            self.log_probs,
            self.advantages,
            self.returns
        )


class ActorNet():
    # def __init__(self, input_dims, lr=0.0003):
    #     # the actor has separate towers for direction and acceleration
    #     # in this way we can train them separately
    #     train_acc = True
    #     train_dir = True
    #
    #     # initializer = tf.keras.initializers.GlorotUniform(seed=None)
    #     initializer = tf.keras.initializers.RandomUniform(minval=.0, maxval=.0005)
    #     inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
    #
    #     # tower of acceleration
    #     out1 = layers.Dense(32, activation="tanh", trainable=train_acc, kernel_initializer=initializer)(inputs)
    #     out1 = layers.Dense(32, activation="tanh", trainable=train_acc, kernel_initializer=initializer)(out1)
    #
    #     # mu,var of accelleration
    #     mu_acc_out = layers.Dense(1, activation='sigmoid', trainable=train_acc, kernel_initializer=initializer)(out1)
    #     var_acc_out = layers.Dense(1, activation='softplus', trainable=train_acc, kernel_initializer=initializer)(out1)
    #
    #     # tower of direction
    #     out2 = layers.Dense(32, activation="tanh", trainable=train_dir, kernel_initializer=initializer)(inputs)
    #     out2 = layers.Dense(32, activation="tanh", trainable=train_dir, kernel_initializer=initializer)(out2)
    #
    #     # mu,var of direction
    #     mu_dir_out = layers.Dense(1, activation='sigmoid', trainable=train_dir, kernel_initializer=initializer)(out2)
    #     var_dir_out = layers.Dense(1, activation='softplus', trainable=train_dir, kernel_initializer=initializer)(out2)
    #     # tanh torna sempre un valore di range fra -1 e +1 ed è inizializzata a 0 come primo valore medio.
    #     # softplus(x) = log(exp(x) + 1) quindi torna un valore compreso nel range [0,inf] e ci fa comodo perchè stima
    #     # la dev standard. Esso viene inizializzato ad 1.
    #
    #     outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])
    #     self.model = keras.Model(inputs, outputs, name="ActorNet")
    #     self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def __init__(self,input_dims, lr):

        initializer =tf.keras.initializers.zeros()

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
        out1 = layers.Dense(32, activation="tanh", kernel_initializer=initializer)(inputs)
        out1 = layers.Dense(32, activation="tanh", kernel_initializer=initializer)(out1)
        out1 = layers.Dense(32, activation="tanh", kernel_initializer=initializer)(out1)
        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='sigmoid',  kernel_initializer=initializer)(out1)
        var_acc_out = layers.Dense(1, activation='softplus', kernel_initializer=initializer)(out1)
        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='sigmoid',  kernel_initializer=initializer)(out1)
        var_dir_out = layers.Dense(1, activation='softplus', kernel_initializer=initializer)(out1)

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])

        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)


    def save_checkpoint(self, path):
        self.model.save(path)


class CriticNet():
    def __init__(self, input_dims, lr):
        # vedi altri initializers su https://keras.io/api/layers/initializers/#henormal-class
        # al link giù ho letto che funziona bene con he_uniform ma non va oltre i 250 in realtà
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

        initializer = tf.keras.initializers.zeros()

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
        out = layers.Dense(32, activation="tanh", kernel_initializer=initializer)(inputs)
        out = layers.Dense(32, activation="tanh", kernel_initializer=initializer)(out)
        outputs = layers.Dense(1, activation="softplus")(out)
        # softplus torna un valore sempre positivo

        self.optimizer = keras.optimizers.Adam(learning_rate=lr )
        self.model = keras.Model(inputs, outputs, name="CriticNet")

    def save_checkpoint(self, path):
        self.model.save(path)


class Agent:

    def __init__(self, state_dimension, num_action, alpha, size_memory, path_saving_model, load_models=False, ):

        lr_actor, lr_critic = alpha

        # self.state_dimension = state_dimension
        self.memory = Memory(size=size_memory, state_dim=state_dimension, num_action=num_action)

        self.actor = ActorNet(input_dims=state_dimension, lr=lr_actor)
        self.actor.model.compile(optimizer=self.actor.optimizer)
        self.actor.model.summary()

        self.critic = CriticNet(input_dims=state_dimension, lr=lr_critic)
        self.critic.model.compile(optimizer=self.critic.optimizer, loss='mse')
        self.critic.model.summary()


        if load_models:
            self.load_models(path_saving_model)

    def save_models(self, path):
        print("SAVING ACTOR")
        self.actor.save_checkpoint(path + "/actor")
        print("SAVING CRITIC")
        self.critic.save_checkpoint(path + "/critic")

    def load_models(self, path):
        print("LOADING ACTOR")
        self.actor.model = keras.models.load_model(path + "/actor")
        print("LOADING CRITIC")
        self.critic.model = keras.models.load_model(path + "/critic")

    def action_sampling(self, acc_distribution,dir_distribution):

        acc_samples, acc_log_probs = acc_distribution.experimental_sample_and_log_prob()
        dir_samples, dir_log_probs = dir_distribution.experimental_sample_and_log_prob()

        samples = acc_samples, dir_samples
        samples = tf.squeeze(samples)
        # samples = tf.convert_to_tensor(())
        log_probs = acc_log_probs + dir_log_probs

        # es di output: [ valore_accellerazione, valore_direzione ], [ valore_logprob_congiunta_azione ]
        return samples, log_probs  # è un float32

    def act(self, state, action=None):
        dists = tf.convert_to_tensor(self.actor.model(state))
        mean_acc, stddev_acc, mean_dir, stddev_dir = tf.split(dists, num_or_size_splits=4, axis=1)

        # reshape for training
        mean_acc = tf.reshape(mean_acc, shape=[-1])
        stddev_acc = tf.reshape(stddev_acc, shape=[-1])
        mean_dir = tf.reshape(mean_dir, shape=[-1])
        stddev_dir = tf.reshape(stddev_dir, shape=[-1])

        tfd = tfp.distributions
        stddev_acc += 1e-10  # to avoid division by 0
        stddev_dir += 1e-10  # to avoid division by 0

        acc_distribution = tfd.TruncatedNormal(loc=mean_acc, scale=stddev_acc, low=-1, high=+1,
                                               validate_args=True, allow_nan_stats=False, name='accelleration_norm')
        dir_distribution = tfd.TruncatedNormal(loc=mean_dir, scale=stddev_dir, low=-30, high=+30,
                                               validate_args=True, allow_nan_stats=False, name='direction_norm')
        if action is None:
            return self.action_sampling(acc_distribution,dir_distribution)


        acc_value,dir_value = tf.split(action,num_or_size_splits=2,axis=1)
        acc_value = tf.reshape(acc_value, shape=[-1])
        dir_value = tf.reshape(dir_value, shape=[-1])

        log_probs = tf.add(acc_distribution.log_prob(acc_value), dir_distribution.log_prob(dir_value))

        return action, log_probs  # è un float32

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)


    def finish_trajectory(self, last_value=0, gamma=0.99, lam=0.95):
        # path_slice define the bounds of a trajectory
        path_slice = slice(self.memory.trajectory_start_index, self.memory.pointer)
        rewards = self.memory.rewards[path_slice]
        values = np.append(self.memory.values[path_slice], last_value)
        dones = self.memory.dones[path_slice]
        returns = []
        gae = 0.
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - int(dones[i])) - values[i]
            gae = delta + (gamma * lam * (1 - int(dones[i]))) * gae
            returns.insert(0, gae + values[i])

        #by Bellman Equation
        adv = np.array(returns) - values[:-1]

        #normalize advantage
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

        # normalize expected returns
        #returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10 )

        self.memory.advantages[path_slice] = adv
        self.memory.returns[path_slice] = returns
        self.memory.trajectory_start_index = self.memory.pointer

    @tf.function
    def train_actor_network(self, states,old_actions, old_probs, advantages, actor_trainable_variables):
        with tf.GradientTape() as tape:  # everithing here is recorded and released then.
            tape.watch(actor_trainable_variables)
            a, new_probs = self.act(states, action=old_actions)
            ratio = tf.exp( new_probs - old_probs )
            min_advantage = tf.where(
                advantages > 0,
                (1 + .02) * advantages,
                (1 - .02) * advantages,
            )
            # min_advantage = tf.clip_by_value(t=ratio, clip_value_min=(1-0.02), clip_value_max=(1+0.02)) * advantages
            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, min_advantage)
            )

        grads = tape.gradient(actor_loss, actor_trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, actor_trainable_variables))

        _, updated_probs = self.act(states, action=old_actions)
        kl = tf.reduce_mean(
            old_probs
            - updated_probs,
        )
        return tf.reduce_sum(kl)

    @tf.function
    def train_critic_network(self, states, returns, critic_trainable_variables):
        with tf.GradientTape() as tape:
            tape.watch(critic_trainable_variables)
            new_vals = tf.convert_to_tensor(self.critic.model(states))
            value_loss = tf.reduce_mean( tf.pow(returns - new_vals, 2) )

        grads = tape.gradient(value_loss, critic_trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, critic_trainable_variables))

    def learn(self, training_iteration, target_kl):
        # get stored elements
        states, actions, log_probs, advantages, returns = self.memory.get()

        states = tf.convert_to_tensor(states) #float32
        actions = tf.convert_to_tensor(actions)  # float32
        old_probs = tf.convert_to_tensor(log_probs)  #float32
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)

        # train actor network
        print("Training actor")
        actor_trainable_variables = self.actor.model.trainable_variables
        for iter in range(training_iteration):
            kl = self.train_actor_network(states,actions, old_probs, advantages, actor_trainable_variables)
            if kl > target_kl:
                print("Early Stopping ! kl: {a} > target_kl : {b} ".format(a=kl, b=target_kl))
                break

        print("Training critic")
        critic_trainable_variables = self.critic.model.trainable_variables
        for iter in range(training_iteration):
            self.train_critic_network(states, returns, critic_trainable_variables)

        print("Mean Returns : ", tf.reduce_mean(returns))
        print("Mean Advantages : ", tf.reduce_mean(advantages))

    def summary(self, n_epoche):
        x = range(n_epoche)
        plt.plot(x, self.actor_loss_accumulator, label="actor_loss_accumulator")
        # naming the x axis
        plt.xlabel('epoch')
        # naming the y axis
        plt.ylabel('actor loss')
        plt.plot(x, self.critic_loss_accumulator, label="critic_loss_accumulxator")
        # show a legend on the plot
        plt.legend()
        # function to show the plot
        plt.show()
