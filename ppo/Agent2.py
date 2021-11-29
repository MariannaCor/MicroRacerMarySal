# IMPORTSsss
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.dones = []
        self.index, self.trajectory_start_index = 0, 0



    def generate_batches(self):
        print("generates bathces ")
        n_states = len(self.states)
        # numpy.arrange([start, ]stop, [step, ], dtype = None) -> numpy.ndarray
        # es np.arange(0,10,2,float)  -> [0. 2. 4. 6. 8.]
        batch_start = np.arange(0, n_states, self.chunk_size)
        print("batch_start = ", batch_start)
        indices = np.arange(n_states, dtype=np.int64)
        print("indices =", indices)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.chunk_size] for i in batch_start]
        print("batches =", batches)

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.log_probs), \
               np.array(self.rewards), \
               np.array(self.values), \
               np.array(self.advantages), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.index += 1

    def clear_memory(self):
        #print("clearing memory")
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.dones = []


    # # mancano advantage e logprobs
    # def summary(self):
    #      sss = ""
    #      for i in range(len(self.states)):
    #          sss += str(i) + " [\n" \
    #               "\t" + str(self.states[i][0]) + ",\n" \
    #               "\t" + str(self.actions[i]) + ",\n" \
    #               "\t" + str( self.rewards[i]) + ",\n" \
    #               "\t" + str(self.values[i]) + ",\n" \
    #               "\t" + str(self.log_probs[i]) + ",\n" \
    #               "\t" + str(
    #              self.dones[i]) + " ]\n\n"
    #      print(sss)



    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.index, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantages),
            np.std(self.advantages),
        )
        self.advantages = (self.advantages - advantage_mean) / advantage_std
        # order is important!: states,actions,dists_probs, rewards, advantages
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.advantages
        )


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
        mu_acc_out = layers.Dense(1, activation='tanh', trainable=train_acc )(out1)
        var_acc_out = layers.Dense(1, activation='softplus', trainable=train_acc )(out1)

        # tower of direction
        out2 = layers.Dense(32, activation="relu", trainable=train_dir, kernel_initializer=initializer)(inputs)
        out2 = layers.Dense(32, activation="relu", trainable=train_dir , kernel_initializer=initializer)(out2)

        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='tanh', trainable=train_dir)(out2)
        var_dir_out = layers.Dense(1, activation='softplus', trainable=train_dir)(out2)

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])

        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def save_checkpoint(self, path):
        self.model.save(path)


class CriticNet():
    def __init__(self, input_dims, lr=0.0003):

        #tf.keras.initializers.HeUniform() migliore fino ad ora arriva a 300ep
        initializer = tf.keras.initializers.HeNormal()
        #initializer =  tf.keras.initializers.RandomUniform(minval=0.00005, maxval=0.07
        #initializer =  tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.5)
        #initializer = tf.keras.initializers.HeUniform()
        #vedi altri su https://keras.io/api/layers/initializers/#henormal-class
        #al link giù ho letto che funziona bene con he uniform ma non va oltre i 250 in realtà
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

        inputs = keras.Input(shape=[input_dims, ], dtype=tf.float32)

        out = layers.Dense(32, activation="relu",kernel_initializer=initializer)(inputs)
        out = layers.Dense(32, activation="relu",kernel_initializer=initializer)(out)
        out = layers.Dense(32, activation="relu",kernel_initializer=initializer)(out)
        outputs = layers.Dense(1, activation="relu")(out)

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipvalue=0.5)

        self.model = keras.Model(inputs, outputs, name="CriticNet")

    def save_checkpoint(self,path):
        self.model.save(path)



class Agent2:

    def __init__(self, state_dimension, path_saving_model ,alpha=3e-4, load_models=False, ):

        lrActor,lrCritic = alpha
        self.state_dimension = state_dimension
        self.memory = Memory()

        self.actor = ActorNet(input_dims=state_dimension, lr=lrActor)
        self.actor.model.compile(optimizer=self.actor.optimizer)

        self.critic = CriticNet(input_dims=state_dimension, lr=lrCritic)
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


    #TODO: verificare valore noise, troppo grande?
    def get_truncated_normal(self, mean, sd, low=-1, upp=1, noise=0.0005):
        t = tf.random.normal([], mean=mean, stddev=sd, dtype="float32")
        return tf.clip_by_value(t, clip_value_min=(low + noise),clip_value_max=(upp - noise))
        # return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def samplingAction(self, distributions):
        # splitto l'output della rete per colonne, le metto in vettori riga con reshape e prendo i valori di media ed accellerazione prodotti dalla rete nell'ordine
        mean_acc, stddev_acc, mean_dir, stddev_dir = tf.split(distributions, num_or_size_splits=4, axis=1)
        mean_acc = tf.reshape(mean_acc, shape=[-1])
        stddev_acc = tf.reshape(stddev_acc, shape=[-1])
        mean_dir = tf.reshape(mean_dir, shape=[-1])
        stddev_dir = tf.reshape(stddev_dir, shape=[-1])

        # campiono dalle distribuzioni,  nei range un possibile valore per accellerazione e direzione
        acc_sample = self.get_truncated_normal(mean=mean_acc, sd=stddev_acc, low=-1, upp=1)
        dir_sample = self.get_truncated_normal(mean=mean_dir, sd=stddev_dir, low=-30, upp=30)
        samples = acc_sample, dir_sample

        # E poi ottengo la sua prob con la pdf della norma!
        acc_prob = self.pdf(acc_sample, mean=mean_acc, sd=stddev_acc)
        dir_prob = self.pdf(dir_sample, mean=mean_dir, sd=stddev_dir)
        log_probs = tf.math.log(acc_prob) + tf.math.log(dir_prob)  # log di prob congiunta fra i due eventi indipendenti.
        log_probs = tf.reshape(log_probs, shape=[-1])
        samples = tf.squeeze(samples)
        # es di output: [ valore_accellerazione, valore_direzione ], [ valore_logprob_congiunta_azione ]
        ret = samples, log_probs
        return ret  #è un float32

    def act(self, state):
        dists = tf.convert_to_tensor ( self.actor.model(state) )
        if np.isnan(dists).any():
            print(dists)
            sys.exit("Errore tornato nan dalla neural network actor ")

        ret = self.samplingAction(dists)

        return ret

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)

    def clean_memory(self):
        self.memory.clear_memory()

    # def summary(self):
    #     print("The Agent is now working:\n")
    #     print("memory\n")
    #     self.memory.summary()
    #     plt.plot(self.actor_loss_accumulator, label="actor_loss_accumulator")
    #     plt.plot(self.critic_loss_accumulator, label="critic_loss_accumulator")
    #     # show a legend on the plot
    #     plt.legend()
    #     # function to show the plot
    #     plt.show()
    #
    #     # ..other information will be added later
    #     # self.actor.summary()


    #@tf.function
    def calculate_advantages(self, last_value=0, gamma=0.99, lam=0.95):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.memory.trajectory_start_index, self.memory.index)
        rewards = np.append(self.memory.rewards[path_slice], last_value)
        values = np.append(self.memory.values[path_slice], last_value)
        #print("rewards", rewards)
        #print("values", values)
        # è una lista di tutti gli delta-i consiste in una serie di passate dall'elemento i fino alla fine dell'array.
        deltas = rewards[:-1] + (gamma * values[1:]) - values[:-1]
        # print("DELTAS = ", deltas)
        self.memory.advantages[path_slice] = self.discounted_cumulative_sums(deltas, gamma * lam)
        self.memory.rewards[path_slice] = self.discounted_cumulative_sums(rewards, gamma)[:-1]
        self.memorytrajectory_start_index = self.memory.index

    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


    def train_actor_network(self, states, old_probs, advantages,actor_trainable_variables):

        #print("TRAINING Actor Net ", str(iter + 1))
        with tf.GradientTape() as tape:  # everithing here is recorded and released them.
            tape.watch(actor_trainable_variables)
            # forward step - TODO: seconda passata è corretto avere un altro campionamento?
            # print("states in input => ",states)
            if np.isnan(states).any():
                print(states)
                sys.exit("ACTOR :: states is nan exiting ...")
            _, new_probs = self.act(states)

            # print("new_probs ", new_probs)
            # compute loss
            prob_ratio = tf.math.divide(tf.exp(new_probs), tf.exp(old_probs))
            if np.isnan(prob_ratio).any():
                print(prob_ratio)
                sys.exit("ACTOR :: prob_ratio is nan exiting ...")

            min_advantage = tf.where(
                advantages > 0,
                (1 + .2) * advantages,
                (1 - .2) * advantages,
            )
            min_advantage = tf.cast(min_advantage, tf.float32)
            advantages = tf.cast(advantages, tf.float32)
            partial = tf.math.multiply(prob_ratio, advantages)
            # print("#p1 ", partial)
            partial = tf.math.minimum(partial, min_advantage)
            #print("#p2 ", partial)
            partial = tf.reduce_mean(partial)
            #print("#p3 ", partial)
            actor_loss = tf.math.negative(partial)
            #print("A ", actor_loss)

        grads = tape.gradient(actor_loss, actor_trainable_variables)
        # print("grads len = ", len( grads))
        #print("grads ACTOR= ", grads)
        self.actor.optimizer.apply_gradients(zip(grads, actor_trainable_variables))


    def train_critic_network(self,states, rewards,critic_trainable_variables):
        with tf.GradientTape() as tape:
            tape.watch(critic_trainable_variables)

            if np.isnan(states).any():
                print(states)
                sys.exit("CRITIC :: states is nan exiting ...")

            xx = self.critic.model(states)
            if (np.isnan(xx)).any():
                print("xx is nan ? ", xx)
                sys.exit("CRITIC :: xx is nan exiting ...")

            new_vals = tf.convert_to_tensor(xx)
            #print("new vals ",new_vals)

            # forward and loss calculation
            value_loss = tf.reduce_mean((rewards - new_vals) ** 2)
            #print(" V ",value_loss)

        # print("value _loss ", value_loss)
        grads = tape.gradient(value_loss, critic_trainable_variables)
        # print("grads len = ", len(grads))
        #print("grads CRITIC ", grads)
        self.critic.optimizer.apply_gradients(zip(grads, critic_trainable_variables))


    def learn(self, training_iteration):
        # get stored elements
        states, actions, dists_probs, rewards, advantages = self.memory.get()

        # some preprocessing of saved data
        # expected a tensor like this: tf.Tensor([ [], [], ... )
        states = tf.squeeze(tf.convert_to_tensor(states),
                            axis=1)  # cosi puoi dare in pasto alla rete tutto l'array di stati in un unico colpo.
        old_probs = tf.convert_to_tensor(dists_probs)
        actor_trainable_variables = self.actor.model.trainable_variables

        # train actor network
        for iter in range(training_iteration):
            self.train_actor_network(states,old_probs,advantages,actor_trainable_variables)

        critic_trainable_variables = self.critic.model.trainable_variables
        for iter in range(training_iteration):
            self.train_critic_network(states,rewards,critic_trainable_variables)


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



