# IMPORTSsss
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from tensorflow.keras import layers



class Memory:
    def __init__(self,size, state_dim, num_action):
        #shape = (col,righe) or (righe,colonne), size deve essere il numero di colonne ...  ?
        self.states  = np.zeros(shape=(size,state_dim) , dtype=np.float32)
        self.actions = np.zeros(shape=(size,num_action) , dtype=np.float32)
        self.log_probs = np.zeros(size , dtype=np.float32)
        self.rewards = np.zeros(size , dtype=np.float32)    #quella che ritorna l'environment
        self.values = np.zeros(size , dtype=np.float32)     #quella che ritorna il critico
        self.dones = np.zeros(size , dtype=np.float32)      #se l'episodio è finito

        self.advantages= np.zeros(size , dtype=np.float32) #quella che mi serve per calcolare l'actor_loss
        self.returns  = np.zeros(size , dtype=np.float32) #quella che mi serve per calcolare il critic loss

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
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = np.mean(self.advantages)
        advantage_std = np.std(self.advantages)
        self.advantages = (self.advantages - advantage_mean) / advantage_std

        return (
            self.states,
            self.log_probs,
            self.advantages,
            self.returns
        )

    # def get(self):
    #     # Get all data of the buffer and normalize the advantages
    #     self.index, self.trajectory_start_index = 0, 0
    #     advantage_mean, advantage_std = (
    #         np.mean(self.advantages),
    #         np.std(self.advantages),
    #     )
    #     self.advantages = (self.advantages - advantage_mean) / advantage_std
    #     # order is important!: states,actions,dists_probs, rewards, advantages
    #     return (
    #         self.states,
    #         self.actions,
    #         self.log_probs,
    #         self.rewards,
    #         self.values,
    #         self.advantages
    #     )


class ActorNet():

    def __init__(self, input_dims, lr=0.0003):
        # the actor has separate towers for direction and acceleration
        # in this way we can train them separately
        train_acc = True
        train_dir = True

       # initializer = tf.keras.initializers.GlorotUniform(seed=None)
        initializer = tf.keras.initializers.HeUniform()

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)

        # tower of acceleration
        out1 = layers.Dense(32, activation="relu", trainable=train_acc, kernel_initializer=initializer )(inputs)
        out1 = layers.Dense(32, activation="relu", trainable=train_acc, kernel_initializer=initializer )(out1)

        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='softplus', trainable=train_acc, kernel_initializer=tf.keras.initializers.zeros() )(out1)
        var_acc_out = layers.Dense(1, activation='relu', trainable=train_acc,kernel_initializer=tf.keras.initializers.ones() )(out1)
        #tanh torna sempre un valore di range fra -1 e +1 ed è inizializzata a 0 come primo valore medio.
        #softplus(x) = log(exp(x) + 1) quindi torna un valore compreso nel range [0,inf] e ci fa comodo perchè stima
        #la dev standard. Esso viene inizializzato ad 1.

        # tower of direction
        out2 = layers.Dense(32, activation="relu", trainable=train_dir, kernel_initializer=initializer)(inputs)
        out2 = layers.Dense(32, activation="relu", trainable=train_dir , kernel_initializer=initializer)(out2)

        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='softplus', trainable=train_dir, kernel_initializer=tf.keras.initializers.zeros() )(out2)
        var_dir_out = layers.Dense(1, activation='relu', trainable=train_dir, kernel_initializer=tf.keras.initializers.ones() )(out2)
        #tanh torna sempre un valore di range fra -1 e +1 ed è inizializzata a 0 come primo valore medio.
        #softplus(x) = log(exp(x) + 1) quindi torna un valore compreso nel range [0,inf] e ci fa comodo perchè stima
        #la dev standard. Esso viene inizializzato ad 1.

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])
        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr,clipvalue= 0.5 )

    def save_checkpoint(self, path):
        self.model.save(path)


class CriticNet():
    def __init__(self, input_dims, lr=0.0003):

        #vedi altri initializers su https://keras.io/api/layers/initializers/#henormal-class
        #al link giù ho letto che funziona bene con he_uniform ma non va oltre i 250 in realtà
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        initializer = tf.keras.initializers.HeUniform()

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)
        out = layers.Dense(32, activation="tanh",kernel_initializer=initializer)(inputs)
        out = layers.Dense(32, activation="tanh", kernel_initializer=initializer )(out)
        outputs = layers.Dense(1, activation="softplus")(out)
        #softplus torna un valore

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipvalue= 0.5 )
        self.model = keras.Model(inputs, outputs, name="CriticNet")

    def save_checkpoint(self,path):
        self.model.save(path)



class Agent2:

    def __init__(self, state_dimension, num_action, alpha, size_memory, path_saving_model ,load_models=False , ):

        lr_actor , lr_critic  = alpha

        #self.state_dimension = state_dimension
        self.memory = Memory(size=size_memory, state_dim=state_dimension, num_action=num_action)

        self.actor = ActorNet(input_dims=state_dimension, lr=lr_actor)
        self.actor.model.compile(optimizer=self.actor.optimizer)

        self.critic = CriticNet(input_dims=state_dimension, lr=lr_critic)
        self.critic.model.compile(optimizer=self.critic.optimizer)

        if load_models:
            self.load_models(path_saving_model)


    def pdf(self, guess, mean, sd):
        return 1 / tf.math.sqrt(2 * math.pi) * tf.math.exp((-guess ** 2) / 2)

    def save_models(self,path):
        print("SAVING ACTOR")
        self.actor.save_checkpoint(path+"/actor")
        print("SAVING CRITIC")
        self.critic.save_checkpoint(path+"/critic")

    def load_models(self,path):
        print("LOADING ACTOR")
        self.actor.model = keras.models.load_model(path+"/actor" )
        print("LOADING CRITIC")
        self.critic.model= keras.models.load_model(path+"/critic")

    def action_sampling(self, distributions):

        def sample_and_log_prob(dist, up, down):
            samples= dist.sample()
            up = tf.convert_to_tensor(up, dtype="float32")
            down = tf.convert_to_tensor(down, dtype="float32")
            print("\nGot ",samples)
            if len(samples) > 1:
                print("apply on vectors")
                tensor = tf.div(
                    tf.subtract(
                        samples,
                        tf.reduce_min(samples)
                    ),
                    tf.subtract(
                        tf.reduce_max(samples),
                        tf.reduce_min(samples)
                    )
                )
            else:
                print("apply formula")
                tensor = tf.subtract(samples ,down ) / tf.subtract(up , down)

            normalized_value = tensor
            acceptance = (down < normalized_value< up )
            print("{} first then {} is it in the range ? {} - {} = > {}".format(samples,normalized_value,down,up,acceptance))



            # accepted = False
            # print("Is {} accepted ? {} ".format(samples, accepted))
            # while not accepted:
            #     cond1 = tf.less_equal(samples, up)      # sample < up
            #     cond2 = tf.greater_equal(samples, down) # sampe > down
            #     accepted = tf.logical_and(cond1, cond2) #if down < sample < up
            #     samples = tf.where(tf.logical_not(accepted) , samples, dist.sample())
            #     print("Is {} accepted ? {} ".format(samples, accepted))

            return samples, dist.log_prob(normalized_value)


        # splitto l'output della rete per colonne, le metto in vettori riga con reshape e prendo i valori di media ed accellerazione prodotti dalla rete nell'ordine
        mean_acc, stddev_acc, mean_dir, stddev_dir = tf.split(distributions, num_or_size_splits=4, axis=1)
        #necessary for training
        mean_acc = tf.reshape(mean_acc, shape=[-1])
        stddev_acc = tf.reshape(stddev_acc, shape=[-1])
        mean_dir = tf.reshape(mean_dir, shape=[-1])
        stddev_dir = tf.reshape(stddev_dir, shape=[-1])

        if len(mean_acc) > 1:
            print("mean_acc ", mean_acc)
            print("stddev_acc ", stddev_acc)
            print("mean_dir ", mean_dir)
            print("stddev_dir ", stddev_dir)

        tfd = tfp.distributions
        acc_distribution = tfd.Normal(loc=mean_acc , scale=stddev_acc, validate_args=True, allow_nan_stats=False)
        dir_distribution = tfd.Normal(loc=mean_dir, scale=stddev_dir, validate_args=True, allow_nan_stats=False)


        acc_samples ,acc_log_probs = sample_and_log_prob(acc_distribution, up=1, down=(-1) )
        dir_samples ,dir_log_probs = sample_and_log_prob(dir_distribution, up=+30, down=(-30))

        samples = acc_samples,dir_samples
        log_probs = acc_log_probs + dir_log_probs
        return samples, log_probs

        # campiono dalle distribuzioni,  nei range un possibile valore per accellerazione e direzione

        # TODO: verificare valore noise, troppo grande?
        # acc_sample = sampling_values_in_a_range(acc_distribution, low=-1, up=1)
        # dir_sample = sampling_values_in_a_range(dir_distribution, low=-30, up=30)
        # samples = acc_sample, dir_sample
        #
        # acc_log_prob = acc_distribution.log_prob(acc_sample)
        # dir_log_prob = dir_distribution.log_prob(dir_sample)
        #
        # log_probs = acc_log_prob + dir_log_prob  # log di prob congiunta fra i due eventi indipendenti
        #log_probs = tf.reshape(log_probs, shape=[-1])

        #samples = tf.squeeze(samples)
        # es di output: [ valore_accellerazione, valore_direzione ], [ valore_logprob_congiunta_azione ]
       # ret = samples, log_probs
       # return ret  #è un float32

    def act(self, state):
        dists = tf.convert_to_tensor ( self.actor.model(state) )
        if(len(dists)> 1):
            print("dists ",dists)
        ret = self.action_sampling(dists)
        return ret

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)

    def clean_memory(self):
        self.memory.clear_memory()



    def finish_trajectory(self, last_value=0 ,gamma=0.99, lam=0.95):

        #we declare this inner function to reuse it two times.
        def discounted_cumulative_sums(x, discount):
            # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        # Finish the trajectory by computing advantage estimates and rewards-to-go to
        path_slice = slice(self.memory.trajectory_start_index, self.memory.pointer)
        rewards = np.append(self.memory.rewards[path_slice], last_value)
        values = np.append(self.memory.values[path_slice], last_value)
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        self.memory.advantages[path_slice] = discounted_cumulative_sums(deltas, gamma * lam)
        self.memory.returns[path_slice] = discounted_cumulative_sums(rewards, gamma )[:-1]
        self.memory.trajectory_start_index = self.memory.pointer


   # @tf.function
    def train_actor_network(self, states, old_probs, advantages,actor_trainable_variables):

        with tf.GradientTape() as tape:  # everithing here is recorded and released them.
            tape.watch(actor_trainable_variables)
            # forward step - TODO: seconda passata è corretto avere un altro campionamento?
            a, new_probs = self.act(states)
            print("{} and log_probs {}".format(a,new_probs))
            # compute loss
            prob_ratio = tf.math.exp(new_probs - old_probs )

            min_advantage = tf.where(
                advantages > 0,
                (1 + .2) * advantages,
                (1 - .2) * advantages,
            )
            partial = tf.math.multiply(prob_ratio, advantages)
            partial = tf.math.minimum(partial, min_advantage)
            partial = tf.reduce_mean(partial)
            actor_loss = tf.math.negative(partial)
            print("actor_loss = ",actor_loss)

        grads = tape.gradient(actor_loss, actor_trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, actor_trainable_variables))

        _,updated_probs = self.act(states)
        kl = tf.reduce_mean(
            old_probs
            - updated_probs,
        )
        kl = tf.reduce_sum(kl)
        return kl

    #@tf.function
    def train_critic_network(self,states, returns ,critic_trainable_variables):
        with tf.GradientTape() as tape:
            tape.watch(critic_trainable_variables)
            new_vals = tf.convert_to_tensor(self.critic.model(states))
            value_loss = tf.reduce_mean((returns - new_vals) ** 2)
            print("critc_loss = ", value_loss)

        grads = tape.gradient(value_loss, critic_trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, critic_trainable_variables))


    def learn(self, training_iteration,target_kl):
        # get stored elements
        states, log_probs, advantages ,returns = self.memory.get()
        # some preprocessing of saved data
        # expected tensors like this: tf.Tensor([ [], [], ... )
        states = tf.convert_to_tensor(states)  # float32
        old_probs = tf.convert_to_tensor(log_probs ) #float32

        # train actor network
        print("Training actor")
        actor_trainable_variables = self.actor.model.trainable_variables

        for iter in range(training_iteration):

            kl = self.train_actor_network( states, old_probs, advantages, actor_trainable_variables )
            if kl > target_kl:
                # Early Stopping
                print("Early Stopping")
                print("kl = {a} > {b} ".format(a=kl,b=target_kl) )
                break


        print("Training critic")
        critic_trainable_variables = self.critic.model.trainable_variables
        print("returns values are ",returns)
        for iter in range(training_iteration):
            self.train_critic_network(states,returns,critic_trainable_variables)



    def summary(self,n_epoche):
        x = range(n_epoche)
        plt.plot(x,self.actor_loss_accumulator, label="actor_loss_accumulator")
        # naming the x axis
        plt.xlabel('epoch')
        # naming the y axis
        plt.ylabel('actor loss')
        plt.plot(x,self.critic_loss_accumulator, label="critic_loss_accumulxator")
        # show a legend on the plot
        plt.legend()
        # function to show the plot
        plt.show()



