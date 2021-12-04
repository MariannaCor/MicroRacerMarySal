# IMPORTSsss
import math
import sys
import numpy as np


import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

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
        initializer =  tf.keras.initializers.RandomUniform(minval=0., maxval=.0005)

        inputs = keras.Input(shape=[input_dims,], dtype=tf.float32)

        # tower of acceleration
        out1 = layers.Dense(32, activation="tanh", trainable=train_acc, kernel_initializer=initializer )(inputs)
        out1 = layers.Dense(32, activation="tanh", trainable=train_acc, kernel_initializer=initializer )(out1)

        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='sigmoid', trainable=train_acc, kernel_initializer=initializer )(out1)
        var_acc_out = layers.Dense(1, activation='softplus', trainable=train_acc,kernel_initializer=initializer )(out1)

        # tower of direction
        out2 = layers.Dense(32, activation="tanh", trainable=train_dir, kernel_initializer=initializer)(inputs)
        out2 = layers.Dense(32, activation="tanh", trainable=train_dir , kernel_initializer=initializer)(out2)

        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='sigmoid', trainable=train_dir, kernel_initializer=initializer )(out2)
        var_dir_out = layers.Dense(1, activation='softplus', trainable=train_dir, kernel_initializer=initializer )(out2)
        #tanh torna sempre un valore di range fra -1 e +1 ed è inizializzata a 0 come primo valore medio.
        #softplus(x) = log(exp(x) + 1) quindi torna un valore compreso nel range [0,inf] e ci fa comodo perchè stima
        #la dev standard. Esso viene inizializzato ad 1.

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])
        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr )

    def save_checkpoint(self, path):
        self.model.save(path)


class CriticNet():
    def __init__(self, input_dims, lr=0.0003):

        #vedi altri initializers su https://keras.io/api/layers/initializers/#henormal-class
        #al link giù ho letto che funziona bene con he_uniform ma non va oltre i 250 in realtà
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        initializer = tf.keras.initializers.zeros()

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
        # splitto l'output della rete per colonne, le metto in vettori riga con reshape e prendo i valori di media ed accellerazione prodotti dalla rete nell'ordine
        mean_acc, stddev_acc, mean_dir, stddev_dir = tf.split(distributions, num_or_size_splits=4, axis=1)

        #reshape for training
        mean_acc = tf.reshape(mean_acc, shape=[-1])
        stddev_acc = tf.reshape(stddev_acc, shape=[-1])
        mean_dir = tf.reshape(mean_dir, shape=[-1])
        stddev_dir = tf.reshape(stddev_dir, shape=[-1])

        tfd = tfp.distributions
        stddev_acc +=0.00000005
        stddev_dir += 0.00000005

        acc_distribution = tfd.TruncatedNormal(loc=mean_acc , scale=stddev_acc, low=-1, high=+1, validate_args=True, allow_nan_stats=False,name='accelleration_norm')
        dir_distribution = tfd.TruncatedNormal(loc=mean_dir, scale=stddev_dir, low=-30, high=+30, validate_args=True, allow_nan_stats=False,name='direction_norm')

        acc_samples, acc_log_probs = acc_distribution.experimental_sample_and_log_prob()
        dir_samples,dir_log_probs = dir_distribution.experimental_sample_and_log_prob()

        # chek1 = tf.where( tf.logical_and( tf.less_equal(acc_samples,-1), tf.greater_equal(acc_samples,+1)))
        #
        # if len(chek1)> 0:
        #     print(chek1)
        #     sys.exit("prob with sampling of accelleration")
        #
        # chek1 = tf.where( tf.logical_and( tf.less_equal(dir_samples,-30), tf.greater_equal(dir_samples,+30)))
        # if len(chek1)>0 :
        #     print(chek1)
        #     sys.exit("prob with sampling of direction")


        samples = acc_samples,dir_samples
        samples = tf.squeeze(samples)
        #samples = tf.convert_to_tensor( ())
        log_probs = acc_log_probs + dir_log_probs
        # es di output: [ valore_accellerazione, valore_direzione ], [ valore_logprob_congiunta_azione ]
        return samples, log_probs #è un float32

    def act(self, state):
        dists = tf.convert_to_tensor ( self.actor.model(state) )
        return self.action_sampling(dists)

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)

    def finish_trajectory(self, last_value=0, gamma=0.99, lam=0.95):
        path_slice = slice(self.memory.trajectory_start_index, self.memory.pointer)
        rewards = self.memory.rewards[path_slice]
        values = np.append(self.memory.values[path_slice], last_value)
        dones =  self.memory.dones[path_slice]
        returns = []
        gae = 0.
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i+1] * (1 - int(dones[i])) - values[i]
            gae = delta + (gamma*lam) * (1-int(dones[i])) * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        adv = (adv - np.mean(adv) ) / (np.std(adv) + 1e-10 )

        self.memory.advantages[path_slice] = adv

        self.memory.returns[path_slice] = returns
        self.memory.trajectory_start_index = self.memory.pointer

        # discount = 1
        # a_t = 0
        # for i in range(len(rewards)):
        #     delta = rewards[i] + gamma * values[i] * (1 - dones[i]) - values[i+1]
        #     a_t += discount * delta
        #     discount *= gamma * lam
        # advantages[i] = a_t

        #deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        #deltas = rewards[:-1] + gamma * values[1:] * (1 - dones[1:]) - values[:-1]
        #self.memory.advantages[path_slice] = discounted_cumulative_sums(deltas, gamma * lam)
        #self.memory.returns[path_slice] = discounted_cumulative_sums(rewards, gamma )[:-1]
        #self.memory.returns[path_slice] =  self.memory.advantages[path_slice] + self.memory.values[path_slice]




    @tf.function
    def train_actor_network(self, states, old_probs, advantages,actor_trainable_variables):
        with tf.GradientTape() as tape:  # everithing here is recorded and released then.
            tape.watch(actor_trainable_variables)
            # forward step - TODO: seconda passata è corretto avere un altro campionamento?
            a, new_probs = self.act(states)
            #print("{} and log_probs {}".format(a,new_probs))
            # compute loss
            prob_ratio = tf.math.exp(new_probs - old_probs )
            clip_value = .2
            clip_probs = tf.clip_by_value(prob_ratio, 1.-clip_value, 1.+clip_value)
            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    tf.multiply(prob_ratio, advantages),
                    tf.multiply(clip_probs, advantages)
                )
            )
            #print("actor_loss = ",actor_loss)
            # partial1 = tf.math.multiply(prob_ratio, advantages)
            # clip_value= 0.2
            # clip_probs = tf.clip_by_value(prob_ratio, 1.-clip_value, 1.+ clip_value)
            # partial2 = tf.multiply(clip_probs,advantages)
            #
            # #min_advantage = tf.where(
            # #    advantages > 0,
            # #    (1 + .2) * advantages,
            # #    (1 - .2) * advantages,
            # #)
            #
            # loss = tf.math.minimum(partial1, partial2)
            # actor_loss = -tf.reduce_mean(loss)
            #print("actor_loss = ",actor_loss)

        grads = tape.gradient(actor_loss, actor_trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, actor_trainable_variables))

        _,updated_probs = self.act(states)
        kl = tf.reduce_mean(
             old_probs
             - updated_probs,
        )
        return tf.reduce_sum(kl)

    @tf.function
    def train_critic_network(self,states, returns ,critic_trainable_variables):
        with tf.GradientTape() as tape:
            tape.watch(critic_trainable_variables)
            new_vals = tf.convert_to_tensor(self.critic.model(states))
            #squared error loss
            value_loss = tf.reduce_mean((returns - new_vals) ** 2)
            #print("critic_loss ",value_loss)

        grads = tape.gradient(value_loss, critic_trainable_variables)
        self.critic.optimizer.apply_gradients(zip(grads, critic_trainable_variables))



    def learn(self, training_iteration,target_kl):
        # get stored elements
        states, log_probs, advantages ,returns = self.memory.get()

        states = tf.convert_to_tensor(states)# float32
        old_probs = tf.convert_to_tensor(log_probs)# float32
        advantages = tf.convert_to_tensor(advantages)
        returns = tf.convert_to_tensor(returns)

        # train actor network
        print("Training actor")
        actor_trainable_variables = self.actor.model.trainable_variables
        for iter in range(training_iteration):
            kl = self.train_actor_network( states, old_probs, advantages, actor_trainable_variables )
            if kl > target_kl:
               # Early Stopping
               print("Early Stopping ! kl: {a} > target_kl : {b} ".format(a=kl,b=target_kl) )
               break


        print("Training critic")
        critic_trainable_variables = self.critic.model.trainable_variables
        for iter in range(training_iteration):
            self.train_critic_network(states,returns,critic_trainable_variables)

        print("Mean Returns : ", tf.reduce_mean(returns))
        print("Mean Advantages : ", tf.reduce_mean(advantages))


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



