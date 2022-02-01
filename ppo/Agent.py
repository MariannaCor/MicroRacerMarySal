# IMPORTSsss
import math
import sys

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


    def __init__(self,input_dims, lr,train_acc=True, train_dir=True):

        initializer =tf.keras.initializers.RandomUniform(minval=0.003, maxval=+0.003)

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
        acc = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc ,name ="acc_1")(inputs)
        acc = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc, name ="acc_2")(acc)
        acc = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc, name="acc_3")(acc)
        mu_acc_out = layers.Dense(1, activation='tanh',  kernel_initializer=initializer, name="acc_out_mu")(acc)

        dir = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc, name="dir_1")(inputs)
        dir = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc, name="dir_2")(dir)
        dir = layers.Dense(128, activation="relu", kernel_initializer=initializer, trainable=train_acc, name="dir_3")(dir)
        mu_dir_out = layers.Dense(1, activation='tanh',  kernel_initializer=initializer, name="dir_out_mu")(dir)

        outputs = layers.concatenate([mu_acc_out,mu_dir_out])

        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)


    def save_checkpoint(self, path):
        self.model.save(path)



class CriticNet():
    def __init__(self, input_dims, lr):
        # vedi altri initializers su https://keras.io/api/layers/initializers/#henormal-class
        # al link giù ho letto che funziona bene con he_uniform ma non va oltre i 250 in realtà
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

        initializer =tf.keras.initializers.RandomUniform(minval=0.003, maxval=+0.003)

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
        out = layers.Dense(128, activation="relu", kernel_initializer=initializer, name ="layer_1")(inputs)
        out = layers.Dense(128, activation="relu", kernel_initializer=initializer, name ="layer_2")(out)
        out = layers.Dense(128, activation="relu", kernel_initializer=initializer, name ="layer_3")(out)
        outputs = layers.Dense(1, activation="tanh")(out) #softplus torna un valore sempre positivo.

        self.optimizer = keras.optimizers.Adam(learning_rate=lr )
        self.model = keras.Model(inputs, outputs, name="CriticNet")

    def save_checkpoint(self, path):
        self.model.save(path)


class Agent:

    def __init__(self, state_dimension, num_action, alpha, size_memory, path_saving_model, load_models=False,
                 train_acceleration=True,train_direction = True, clip = .1
    ):
        self.clip_ratio = clip
        self.lr_actor, self.lr_critic = alpha
        self.num_action = num_action
        self.memory = Memory(size=size_memory, state_dim=state_dimension, num_action=num_action)
        self.actor = ActorNet(input_dims=state_dimension, lr=self.lr_actor, train_acc=train_acceleration,train_dir=train_direction)
        self.critic = CriticNet(input_dims=state_dimension, lr=self.lr_critic)

        if load_models:
            self.load_models(path_saving_model)
            self.actor.model.trainable = True
            self.critic.model.trainable = True
            #per tutti i livelli interni metto i valori di train_direction e accelleration dati.
            # 1,3,5 sono i layers di accellerazione
            for layer in self.actor.model.layers:
                if layer.name[0:3] == "acc":
                    layer.trainable = train_acceleration
                if layer.name[0:3] == "dir":
                    layer.trainable = train_direction


        # Don't forget to re-compile the model
        self.actor.model.compile(optimizer=self.actor.optimizer)
        self.critic.model.compile(optimizer=self.critic.optimizer, loss='mse')

        print("CURRENT ACTOR-CRITC NETWORK IN USE")
        self.actor.model.summary()
        self.critic.model.summary()


    def save_models(self, path):
        #print("saving NN to: {}".format(path))
        self.actor.save_checkpoint(path + "/actor")
        self.critic.save_checkpoint(path + "/critic")

    def load_models(self, path):
        print("loading NN from: {}".format(path))
        self.actor.model = keras.models.load_model(path + "/actor")
        self.critic.model = keras.models.load_model(path + "/critic")


    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)


    @tf.function
    def act(self, state, action=None):
        dists = tf.convert_to_tensor(self.actor.model(state))
        #mean_acc, stddev_acc, mean_dir, stddev_dir = tf.split(dists, num_or_size_splits=4, axis=1)
        mean_acc,  mean_dir  = tf.split(dists, num_or_size_splits=2, axis=1)
        # reshaping used in training phase
        mean_acc = tf.reshape(mean_acc, shape=[-1])
        #stddev_acc = tf.reshape(stddev_acc, shape=[-1])
        mean_dir = tf.reshape(mean_dir, shape=[-1])
        #stddev_dir = tf.reshape(stddev_dir, shape=[-1])
        tfd = tfp.distributions

        acc_distribution = tfd.TruncatedNormal(loc=mean_acc, scale=0.05, low=-1, high=+1,
                                               validate_args=True, allow_nan_stats=False, name='accelleration_norm')
        dir_distribution = tfd.TruncatedNormal(loc=mean_dir, scale=0.05, low=-30, high=+30,
                                               validate_args=True, allow_nan_stats=False, name='direction_norm')

        if action is None:
            acc_samples, acc_log_probs = acc_distribution.experimental_sample_and_log_prob()
            dir_samples, dir_log_probs = dir_distribution.experimental_sample_and_log_prob()
            action = tf.squeeze((acc_samples,dir_samples))
            log_probs = acc_log_probs+dir_log_probs

        else:
            # a sort of split by columns
            acc_values, dir_values = tf.split(action,num_or_size_splits=2,axis=1)
            acc_values = tf.reshape(acc_values, shape=[-1])
            dir_values = tf.reshape(dir_values, shape=[-1])
            log_probs = tf.add(acc_distribution.log_prob(acc_values), dir_distribution.log_prob(dir_values))

        # Es di output returned is made of pair: [ accelleration_value, direction_value ], [ sum_value_of_logprobs ]
        return action, log_probs



    def finish_trajectory(self, last_value, gamma, lam):
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
            #print(f"{i}: gae = {gae}")
            #print(f"{i}: values = {values[i]}")
            returns.insert(0, gae + values[i])

        #print(f"returns {returns}")
        #print(f"values[:-1] {values[:-1]}")
        #by Bellman Equation
        adv = np.array(returns) - values[:-1]
        #print(f"adv {adv}")
        # normalize advantage
        #adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        #print(f"normalized adv {adv}")
        # normalize val_target
        #returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        self.memory.advantages[path_slice] = adv
        self.memory.returns[path_slice] = returns
        self.memory.trajectory_start_index = self.memory.pointer


    @tf.function
    def train_actor_network(self, states,old_actions, old_probs, advantages, actor_trainable_variables):
        with tf.GradientTape() as tape:  # everithing here is recorded and released then.
            tape.watch(actor_trainable_variables)
            a, new_probs = self.act(states, action=old_actions)
            clip_ratio = self.clip_ratio
            ratio = tf.exp(new_probs - old_probs )
            surr1 = - advantages * ratio
            #surr2 = advantages * tf.clip_by_value(t=ratio, clip_value_min=1-clip_value, clip_value_max=1+clip_value)
            surr2 = - tf.where(advantages > 0, (1 + clip_ratio) * advantages,(1 - clip_ratio) * advantages)
            actor_loss = tf.reduce_mean( tf.maximum(surr1, surr2) )

        grads = tape.gradient(actor_loss, actor_trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, actor_trainable_variables))

        #Kullback-Leibler divergence estimation
        #calculates a score that measures the divergence of one probability distribution from another.
        _, updated_probs = self.act(states, action=old_actions )
        mean_diff = tf.reduce_mean(old_probs - updated_probs )
        return mean_diff

    @tf.function
    def train_critic_network(self, states, returns, critic_trainable_variables):
        with tf.GradientTape() as tape:
            tape.watch(critic_trainable_variables)
            new_vals = tf.convert_to_tensor(self.critic.model(states))
            #value_loss = mean quadratic error
            value_loss = tf.reduce_mean(tf.pow(returns - new_vals, 2) )

        grads = tape.gradient(value_loss, critic_trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, critic_trainable_variables))

    def learn(self, training_iteration, target_mean_dif):
        # get stored elements
        states, actions, log_probs, advantages, returns = self.memory.get()

        #everything here is a float32
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        old_probs = tf.convert_to_tensor(log_probs)
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)

        # update actor network
        #print("Training actor")
        actor_trainable_variables = self.actor.model.trainable_variables
        for i in range(training_iteration):
             mean_diff = self.train_actor_network(states, actions, old_probs, advantages, actor_trainable_variables)
             if  mean_diff > target_mean_dif:
                #print(f"Early Stop at {i}!")
                break

        # update critic network
        #print("Training critic")
        critic_trainable_variables = self.critic.model.trainable_variables
        for _ in range(training_iteration):
           self.train_critic_network(states, returns, critic_trainable_variables)

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
