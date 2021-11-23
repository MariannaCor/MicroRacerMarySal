# IMPORTSsss
import numpy as np

import scipy.stats as stats
import scipy.signal
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class Memory:
    def __init__(self, chunk_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.dones = []

        self.index, self.trajectory_start_index = 0, 0

        self.chunk_size = chunk_size

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
        print("clearing memory")
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.dones = []

    # mancano advantage e logprobs
    def summary(self):
        sss = ""
        for i in range(len(self.states)):
            sss  +=str(i) + " [\n"\
            "\t" + str(self.states[i][0]) + ",\n"\
            "\t" + str(self.actions[i]) + ",\n"  \
            "\t" + str(self.rewards[i]) + ",\n"  \
            "\t" + str(self.values[i]) +",\n"    \
            "\t" + str(self.log_probs[i]) + ",\n"\
            "\t" + str(self.dones[i]) + " ]\n\n"
        print(sss)

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.index, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantages),
            np.std(self.advantages),
        )
        self.advantages = (self.advantages - advantage_mean) / advantage_std
        return (
            self.states,
            self.actions,
            self.advantages,
            self.rewards,
            self.log_probs
        )

class ActorNet():

    def __init__(self, input_dims, lr=0.0003):
        # mancano da definire checkpoints e device di esecuzione
        # the actor has separate towers for direction and acceleration
        # in this way we can train them separately

        train_acc = True
        train_dir = True

        inputs = keras.Input(shape=(input_dims,),dtype=tf.float32)
        # acceleration

        out1 = layers.Dense(256, activation="relu",trainable=train_acc)(inputs)
        out1 = layers.Dense(256, activation="relu",trainable=train_acc)(out1)
        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='tanh', trainable=train_acc)(out1)
        var_acc_out = layers.Dense(1, activation='softplus', trainable=train_acc)(out1)

        # direction
        out2 = layers.Dense(256, activation="relu", trainable=train_dir)(inputs)
        out2 = layers.Dense(256, activation="relu", trainable=train_dir)(out2)
        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='tanh', trainable=train_dir)(out2)
        var_dir_out = layers.Dense(1, activation='softplus', trainable=train_dir)(out1)

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])

        self.model = keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def save_checkpoint(self): None

    def load_checkpoint(self): None

class CriticNet():
    def __init__(self, input_dims, lr=0.0003):
        # still missing to define the save and device over execute the net: cpu vs gpu for instance.

        # input the state, output the value.
        inputs = layers.Input(shape=(input_dims,))
        out = layers.Dense(86, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1, activation="relu")(out)

        self.model = keras.Model(inputs, outputs, name="CriticNet")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def save_checkpoint(self): None

    def load_checkpoint(self): None




class Agent2:

    def __init__(self, state_dimension, alpha=0.0003, chunk_memory_size=5,num_actions=2):

        self.alpha = alpha
        self.state_dimension = state_dimension
        self.num_action = num_actions
        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3

        self.memory = Memory(chunk_memory_size)
        self.actor = ActorNet(input_dims=state_dimension, lr=alpha)
        self.critic = CriticNet(input_dims=state_dimension, lr=alpha)

    def pdf(self, guess, mean, sd):
        return 1 / np.sqrt(2 * np.pi) * np.exp((-guess ** 2) / 2)

    def get_truncated_normal(self, mean, sd, low=-1, upp=1):
        return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def samplingAction(self, action):
        # splitto l'output della rete
        acc_dist, dir_dist = tf.split(value=action, num_or_size_splits=2, axis=1)

        # prendo i valori di media ed accellerazione prodotti dalla rete
        mean_acc, stddev_acc = tf.squeeze(acc_dist)
        mean_dir, stddev_dir = tf.squeeze(dir_dist)

        acc_sample = self.get_truncated_normal(mean=mean_acc, sd=stddev_acc, low=-1, upp=1).rvs()
        dir_sample = self.get_truncated_normal(mean=mean_dir, sd=stddev_dir, low=-30, upp=30).rvs()

        sampled_actions = acc_sample, dir_sample

        # E poi ottengo la sua prob con la pdf ?!
        acc_prob = self.pdf(acc_sample, mean=mean_acc, sd=stddev_acc)
        dir_prob = self.pdf(dir_sample, mean=mean_dir, sd=stddev_dir)

        log_prob = tf.math.log( acc_prob ) + tf.math.log(dir_prob) #log di prob congiunta fra i due eventi.

        # es di output: [ valore_accellerazione, valore_direzione ], [log di  (prob_accellerazione congiunta a prob_direzione ) ]
        return tf.convert_to_tensor(sampled_actions, dtype="float32"), \
               tf.cast(log_prob, dtype="float32") #è un float64

    def choose_action(self, state):
       output = self.actor.model(state)
       sampled_action, action_log_prob = self.samplingAction(output)

       return sampled_action, action_log_prob

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)

    #
    # def save_models(self):
    #     print('... saving models ...')
    #     self.actor.save_checkpoint()
    #     self.critic.save_checkpoint()
    #
    # def load_models(self):
    #     print('... loading models ...')
    #     self.actor.load_checkpoint()
    #     self.critic.load_checkpoint()

    # def max_lidar(self, observation, angle=np.pi / 3, pins=19):
    #     arg = np.argmax(observation)
    #     dir = -angle / 2 + arg * (angle / (pins - 1))
    #     dist = observation[arg]
    #     if arg == 0:
    #         distl = dist
    #     else:
    #         distl = observation[arg - 1]
    #     if arg == pins - 1:
    #         distr = dist
    #     else:
    #         distr = observation[arg + 1]
    #     return dir, (distl, dist, distr)
    #
    # def observe(self, racer_state):
    #     if racer_state is None:
    #         return np.array([0])  # not used; we could return None
    #     else:
    #         lidar_signal, v = racer_state
    #         dir, (distl, dist, distr) = self.max_lidar(lidar_signal)
    #         return np.array([dir, distl, dist, distr, v])
    #
    # def fromObservationToModelState(self, observation):
    #     state = self.observe(observation)
    #     state = tf.expand_dims(state, 0)
    #     return state
    def training2(self, state, racer, n_epochs=2, steps_per_epoch=5, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                  policy_clip=0.2, train_policy_iterations = 1, train_value_iterations = 3, target_kl = 0.01,
                  ):
        for ep in range(n_epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0
            episode_return = 0
            episode_length = 0
            state_actual = state

            for t in range(steps_per_epoch):

                action, action_log_prob = self.choose_action(state_actual)
                state_new, reward, done = racer.step(action)
                print("new_state ", state_new)
                print("reward ", reward)
                print("done ", done)

                episode_return += reward
                episode_length += 1

                v_value = self.critic.model(state_actual)

                # in teoria basta state, action, rewprd, value_t, logp_t per il training
                self.memory.store_memory(state_actual, action,  action_log_prob, reward, v_value, done)
                self.memory.summary()

                # Update the state
                state_actual = state_new
                state_actual = self.fromObservationToModelState(state_actual)

                # Finish trajectory if reached to a terminal state
                terminal = done

                if terminal or (t == steps_per_epoch - 1):
                    print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
                    last_value = 0 if done else self.critic.model(state_actual)
                    print("reward ====> ",reward)
                    self.memory.calculate_advantages(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    print("NUM_EPISODES INCREMENTED: ", num_episodes)
                    state_actual, episode_return, episode_length = racer.reset(), 0, 0
                    state_actual = self.fromObservationToModelState(state_actual)

            # Get values from the buffer
            states_buffer, actions_buffer,advantages_buffer,rewards_buffer, logprobabilities_buffer = self.memory.get()
            print("len states = ",len(states_buffer) )

            # Update the policy and implement early stopping using KL divergence
            for _ in range(train_policy_iterations):
                print("TRAINING ACTORNET ", str(_))
                old_probs = tf.squeeze(logprobabilities_buffer)
                advantages_buffer = tf.convert_to_tensor(advantages_buffer, dtype="float32")
                new_probs = []
                for state in states_buffer:
                    _, new_prob = self.choose_action(state)
                    new_probs.append(new_prob)
                new_probs = tf.convert_to_tensor(new_probs)

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(old_probs)
                    tape.watch(advantages_buffer)
                    tape.watch(new_probs)
                    # forward : make a prediction and then compute the loss.
                    # compute loss
                    prob_ratio = tf.math.divide(tf.exp(new_probs), tf.exp(old_probs))

                    min_advantage = tf.cast(tf.where(
                        advantages_buffer > 0,
                        (1+.2) * advantages_buffer,
                        (1-.2) * advantages_buffer,
                    ), tf.float32)
                    print("prob ration = ", prob_ratio)
                    print("advantages_buffer = ", advantages_buffer)
                    print("min_advantage = ", min_advantage)
                    loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages_buffer, min_advantage))

                grads = tape.gradient(target=loss, sources=self.actor.model.trainable_variables)
                a = zip(grads, self.actor.model.trainable_variables)
                print("zip len", len(list(a) ))
                self.actor.optimizer.apply_gradients(a)

            # Update the value function
            for _ in range(train_value_iterations):
                print("TRAINING Critic Net ",str(_))
                new_vals = []
                for state in states_buffer:
                    new_vals.append(self.critic.model(state) )
                new_vals = tf.convert_to_tensor(new_vals)

                with tf.GradientTape(persistent=True) as tape:  # Record operations for automatic differentiation.
                    tape.watch(new_vals)
                    value_loss = tf.reduce_mean((rewards_buffer - new_vals) ** 2)

                value_grads = tape.gradient(value_loss, self.critic.model.trainable_variables)
                b = zip(value_grads, self.critic.model.trainable_variables)
                print("zip len", len(list(b)))
                self.critic.optimizer.apply_gradients(b)

            # Print mean return and length for each epoch
            print(" Epoch: ",ep + 1, ". Mean Return: ", sum_return / num_episodes, ". Mean Length: ", sum_length / num_episodes)
            self.memory.clear_memory()

    def summary(self):
        print("The Agent is now working:\n")
        print("memory\n")
        self.memory.summary()
        # ..other information will be added later
        # self.actor.summary()
        def discounted_cumulative_sums(self, x, discount):
            # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def calculate_advantages(self, last_value=0, gamma=0.99, lam=0.95):
            # Finish the trajectory by computing advantage estimates and rewards-to-go
            path_slice = slice(self.memory.trajectory_start_index, self.memory.index)
            rewards = np.append(self.memory.rewards[path_slice], last_value)
            values = np.append(self.memory.values[path_slice], last_value)
            print("rewards", rewards)
            print("values", values)
            # è una lista di tutti gli delta-i consiste in una serie di passate dall'elemento i fino alla fine dell'array.
            deltas = rewards[:-1] + (gamma * values[1:]) - values[:-1]
            #print("DELTAS = ", deltas)
            self.memory.advantages[path_slice] = self.discounted_cumulative_sums(deltas, gamma * lam)
            self.memory.rewards[path_slice] = self.discounted_cumulative_sums(rewards, gamma)[:-1]
            self.memory.memortrajectory_start_index = self.memory.index


    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def learn(self,train_policy_iterations=3, train_value_iterations=3):
        # Get values from the buffer
        states_buffer, actions_buffer, advantages_buffer, rewards_buffer, logprobabilities_buffer = self.memory.get()
        print("number of states in memory = ", len(states_buffer))

        old_probs = tf.constant(tf.squeeze(logprobabilities_buffer))
        advantages_buffer = tf.constant(advantages_buffer)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(states_buffer)
            tape.watch(old_probs)
            tape.watch(advantages_buffer)
            p_ratio= tf.convert_to_tensor(0, dtype="float32")
            final_loss = tf.convert_to_tensor(0, dtype='float32')
            for i in range(len(states_buffer)-1):
                state = states_buffer[i]
                print("state = ",state)
                #forward
                _, new_p = self.choose_action(state)
                old_p = tf.squeeze(old_probs[i])
                new_ratio = tf.exp(new_p - old_p)
                print("p_ratio ",p_ratio)
                print("new_ratio ", new_ratio)

                #questo nuovo razio deve fare media con il precedente.
                p_ratio = (p_ratio + new_ratio)/ 2 #hence make the mean as iteration during computation of the for.
                print("p_ration", p_ratio)
                a_t = tf.cast(advantages_buffer[i], tf.float32)
                min_advantage = tf.cast(tf.where(
                    a_t > 0,
                    (1 + .2) * a_t,
                    (1 - .2) * a_t,
                ), tf.float32)
                print("min_advantage", min_advantage)
                print("advantages_buffer[i]", a_t)
                loss = -1 * tf.minimum( p_ratio * a_t, min_advantage)
                print("loss ",loss)
                final_loss = ((final_loss + loss )/ 2)
        print("final_loss = ",final_loss)
        grads = tape.gradient(target=loss, sources=self.actor.model.trainable_variables)
        print("grads =", grads)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.model.trainable_variables))

        # Update the policy_actor_network for a number of iterations
        # for _ in range(train_policy_iterations):
        #     print("TRAINING ACTORNET ", str(int(_)+1))
        #     old_probs = tf.squeeze(logprobabilities_buffer)
        #     advantages_buffer = tf.convert_to_tensor(advantages_buffer, dtype="float32")
        #     new_probs = []
        #     for state in states_buffer:
        #         _, new_prob = self.choose_action(state)
        #         new_probs.append(new_prob)
        #         new_probs = tf.convert_to_tensor(new_probs)
        #
        #     with tf.GradientTape(persistent=False) as tape:
        #         #devo calcolare al volo le new_probs e poi fare al volo il loss.
        #         # compute loss
        #         prob_ratio = tf.math.divide(tf.exp(new_probs), tf.exp(old_probs))
        #         tape.watch(prob_ratio)
        #
        #         min_advantage = tf.cast(tf.where(
        #             advantages_buffer > 0,
        #             (1 + .2) * advantages_buffer,
        #             (1 - .2) * advantages_buffer,
        #         ), tf.float32)
        #         tape.watch(min_advantage)
        #         loss = -tf.reduce_mean(tf.minimum(prob_ratio * advantages_buffer, min_advantage))
        #         tape.watch(loss)
        #     print("loss => ",loss)
        #     grads = tape.gradient(target=loss, sources=self.actor.model.trainable_variables)
        #     print("grads =",grads)
        #     update = zip(grads, self.actor.model.trainable_variables)
        #     update = list(update)
        #     self.actor.optimizer.apply_gradients(update)

        # Update the value_critic net for a number of iterations
        for _ in range(train_value_iterations):
            print("TRAINING Critic Net ", str(int(_)+1))
            new_vals = []
            for state in states_buffer:
                new_vals.append(self.critic.model(state))
            new_vals = tf.convert_to_tensor(new_vals)

            with tf.GradientTape(persistent=True) as tape:  # Record operations for automatic differentiation.

                tape.watch(new_vals)
                value_loss = tf.reduce_mean((rewards_buffer - new_vals) ** 2)
                tape.watch(value_loss)
            print("tape.watched_variables() ==", tape.watched_variables())
            value_grads = tape.gradient(value_loss, self.critic.model.trainable_variables)
            update = zip(value_grads, self.critic.model.trainable_variables)
            update = list(update)
            self.critic.optimizer.apply_gradients(update)

        self.memory.clear_memory()

    def learn2(self, train_policy_iterations=3, train_value_iterations=3):


        states_buffer, actions_buffer, advantages_buffer, rewards_buffer, logprobabilities_buffer = self.memory.get()
        print("number of states in memory = ", len(states_buffer))


        # Update the policy and implement early stopping using KL divergence
        target_kl = 0.01
        for iteration in range(train_policy_iterations):
           kl = self.train_policy( states_buffer, logprobabilities_buffer, advantages_buffer )
           if kl > 1.5 * target_kl:
               # Early Stopping
               break

        for iteration in range(train_value_iterations):
            self.train_value_function(states_buffer, rewards_buffer)

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self,states_buffer, rewards_buffer ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((rewards_buffer - self.critic.model(states_buffer)) ** 2)

        value_grads = tape.gradient(value_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(value_grads, self.critic.model.trainable_variables))

    @tf.function
    def train_policy(self, states_buffer, log_probs_buffer, advantages_buffer ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.get_logprobs(states_buffer)
                - log_probs_buffer
            )
            min_advantage = tf.where(
                advantages_buffer > 0,
                (1 + .2) * advantages_buffer,
                (1 - .2) * advantages_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages_buffer, min_advantage)
            )

        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            log_probs_buffer
            - self.get_logprobs(states_buffer)
        )

        kl = tf.reduce_sum(kl)
        return kl

    def get_logprobs(self, states_buffer ):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        newProbs = []
        for state in states_buffer:
            _,prob = self.samplingAction(state)
            newProbs.append(prob)

        return tf.convert_to_tensor(newProbs, dtype="float32")
        #tf.one_hot(a, num_actions) questo comando crea una matrice numactions+numactions e mette il valore di 1 negli indici corrispondenti.
        #quindi è una matrice normalizzata in pratica a 0 ed 1 delle prob che moltiplico per un altra delle log probabilities... quindi ?!

        #log_prob = tf.reduce_sum(logprobabilities_all, axis=1) #non tiene conto delle azioni intraprese..

       # return log_prob
    #    for i in range(len(states_buffer)):
    #        #per ogni stato computo un passo del gradiente.
    #        #mi prendo le nuove variabili che diventano costanti.
    #        state = tf.constant(states_buffer[i])
    #        old_prob = tf.constant(tf.squeeze(logprobabilities_buffer[i]))
    #        advantage = tf.cast(advantages_buffer[i], tf.float32)
    #        ratio_accumulator = tf.costant([])
    #        loss_accumulator = tf.costant([])
    #        grads = self.calculate_gradients(state= state, old_prob= old_prob,advantage= advantage, ratio_accumulator = ratio_accumulator,loss_accumulator= loss_accumulator)
    #        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
    #
    # def calculate_gradients(self, state, old_prob, advantage, ratio_accumulator, loss_accumulator):
    #
    #     with tf.GradientTape() as tape:
    #         _,new_prob = self.actor.model(state)
    #         loss_value = self.calculate_loss(new_prob= new_prob,old_prob = old_prob,advantage= advantage)
    #     return tape.gradient(loss_value, self.actor.trainable_variables)
    #
    # def calculate_loss(self, new_prob, old_prob, advantage):
    #     prob_ratio = tf.exp(new_prob - old_prob)
    #     min_advantage = tf.cast(tf.where(
    #         advantage > 0,
    #         (1 + .2) * advantage,
    #         (1 - .2) * advantage,
    #     ), tf.float32)
    #     return -1 * tf.minimum(prob_ratio * advantage, min_advantage)