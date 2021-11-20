# IMPORTSsss
import numpy as np

import scipy.stats as stats
import scipy.signal
import tensorflow as tf
from numpy import sqrt
from tensorflow.keras import layers
from tensorflow.python.training import optimizer

from MicroRacer_Corinaldesi_Fiorilla import tracks


class Memory:
    def __init__(self, chunk_size, gamma=0.99, lam=0.95):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.advantages = []
        self.dones = []
        self.index, self.trajectory_start_index = 0, 0
        self.gamma, self.lam = gamma, lam
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
            sss += str(i) + " [\n" \
                            "\t" + str(self.states[i][0]) + ",\n" \
                                                            "\t" + str(
                self.actions[i]) + ",\n" \
                                   "\t" + str(self.rewards[i]) + ",\n" \
                                                                 "\t" + str(
                self.values[i]) +",\n" +  "\t" + str(self.log_probs[i]) + ",\n" + "\t" + str(self.dones[
                                                         i]) + " ]\n\n"  # mettere tra values e dones: +  "\t" + str(self.advantages[i]) +   ",\n" +  "\t" + str(self.logprobabilities[i])  + ",\n"\

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

    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.index)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        print("DELTAS = ", deltas)
        self.advantages[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.rewards[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        '''
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + gamma * vals_arr[k + 1] * (1 - int(dones_arr[k])) - vals_arr[k])
                discount *= gamma * gae_lambda
            advantage[t] = a_t
        # uscito dal ciclo converto values ed advantage corrispettivi in tensori tensorflow nell'esempio
        advantage = tf.convert_to_tensor(advantage)
        values = tf.convert_to_tensor(vals_arr)
        
        '''

        self.trajectory_start_index = self.index


class ActorNet():

    def __init__(self, input_dims, lr=0.0003):
        # mancano da definire checkpoints e device di esecuzione

        # the actor has separate towers for direction and acceleration
        # in this way we can train them separately
        inputs = layers.Input(shape=(input_dims,))

        # acceleration
        out1 = layers.Dense(256, activation="relu")(inputs)
        out1 = layers.Dense(256, activation="relu")(out1)
        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='tanh')(out1)
        var_acc_out = layers.Dense(1, activation='softplus')(out1)

        # direction
        out2 = layers.Dense(256, activation="relu")(inputs)
        out2 = layers.Dense(256, activation="relu")(out2)
        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='tanh')(out2)
        var_dir_out = layers.Dense(1, activation='softplus')(out1)

        outputs = layers.concatenate([mu_acc_out, var_acc_out, mu_dir_out, var_dir_out])

        self.model = tf.keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def getModel(self):
        return self.model

    def save_checkpoint(self): None

    def load_checkpoint(self): None


class CriticNet():
    def __init__(self, input_dims, lr=0.0003):
        # still missing to define checkpoints and device over execute the net: cpu vs gpu for instance.

        # input the state, output the value.
        inputs = layers.Input(shape=(input_dims,))
        out = layers.Dense(86, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1, activation="relu")(out)

        self.model = tf.keras.Model(inputs, outputs, name="CriticNet")
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    def save_checkpoint(self): None

    def load_checkpoint(self): None

    def getModel(self):
        return self.model


class Agent:

    def __init__(self, state_dimension, alpha=0.0003, chunk_memory_size=5):

        self.alpha = alpha
        self.state_dimension = state_dimension
        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3

        self.memory = Memory(chunk_memory_size)
        self.actor = ActorNet(input_dims=state_dimension, lr=alpha).getModel()
        self.critic = CriticNet(input_dims=state_dimension, lr=alpha).getModel()

        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = tf.optimizers.Adam(learning_rate=self.value_function_learning_rate)

    # def pdfv2(self, guess, mean, sd):return (np.pi * sd) * np.exp(-0.5 * ((guess - mean) / sd) ** 2)

    # def normalize(self, value, min, max ):
    # return [((value - min) / (max - min) ) for value in values]
    # return (value - min) / (max - min)

    def pdf(self, guess, mean, sd):
        return 1 / np.sqrt(2 * np.pi) * np.exp((-guess ** 2) / 2)

    def get_truncated_normal(self, mean, sd, low=-1, upp=1):
        return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def samplingAction(self, action):

        # splitto l'output della rete
        acc_dist, dir_dist = tf.split(value=action, num_or_size_splits=2, axis=1)

        # prendo i valori di media ed accellerazione prodotti dalla rete
        mean_acc, stddev_acc = tf.squeeze(acc_dist).numpy()
        mean_dir, stddev_dir = tf.squeeze(dir_dist).numpy()

        acc_sample = self.get_truncated_normal(mean=mean_acc, sd=stddev_acc, low=-1, upp=1).rvs()
        dir_sample = self.get_truncated_normal(mean=mean_dir, sd=stddev_dir, low=-30, upp=30).rvs()

        sampled_actions = acc_sample, dir_sample

        # E poi ottengo la sua prob con la pdf ?!
        acc_prob = self.pdf(acc_sample, mean=mean_acc, sd=stddev_acc)
        dir_prob = self.pdf(dir_sample, mean=mean_dir, sd=stddev_dir)

        sampled_log_probs = np.log(acc_prob), np.log(dir_prob)

        # es di output: [ valore_accellerazione, valore_direzione ], [prob_accellerazione, prob_direzione]
        return tf.convert_to_tensor(sampled_actions, dtype="float32"), \
               tf.convert_to_tensor(sampled_log_probs, dtype="float32")

    def choose_action(self, state):
        print("STATE IN CHOOSE ACTION = ", state[0])
        output = self.actor(state)
        # poi campiono in base alla loro distribuzione.
        sampled_action, action_log_prob = self.samplingAction(output)
        return sampled_action, action_log_prob

    def remember(self, state, action, prob, reward, value, done):
        self.memory.store_memory(state, action, prob, reward, value, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def training(self, n_epochs=50, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2):
        print("\n\n start learning ..\n\n ")

        # definiamo un numero di epoche di cui vorremmo fare allenamento.
        for _ in range(n_epochs):
            # per ogni epoca generiamo una serie di osservazioni dalla self memory... la politica di generazione sta a noi definire come.
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            # mi sono salvato in memoria tutti i valori del critico man mano nell array vals.
            # mi creo per ogni epoca un array dove mi stimo l'advantage. questa sotto è l'inizializzazione.
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # per ogni reward ottenuta in questo batch, mi calcolo l'advantage al tempo t, il discount per t+1 e
            # li metto dentro il mio array advantage man mano.

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + gamma * vals_arr[k + 1] * (1 - int(dones_arr[k])) - vals_arr[k])
                    discount *= gamma * gae_lambda
                advantage[t] = a_t
            # uscito dal ciclo converto values ed advantage corrispettivi in tensori tensorflow nell'esempio
            advantage = tf.convert_to_tensor(advantage)
            values = tf.convert_to_tensor(vals_arr)

            for batch in batches:
                states = tf.convert_to_tensor(state_arr[batch])
                old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                actions = tf.convert_to_tensor(action_arr[batch])

                outputs = self.actor(states)
                _, new_probs = self.samplingAction(outputs)

                critic_value = self.critic(states)
                critic_value = tf.squeeze(critic_value)

                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio

                weighted_clipped_probs = tf.clip_by_value((prob_ratio, 1 - self.policy_clip,
                                                           1 + self.policy_clip) * advantage[batch])

                actor_loss = -tf.minimum(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # total_loss = actor_loss + 0.5 * critic_loss
                # grads = tf.GradientTape().gradient(actor_loss, self.actor.trainable_variables)
                # self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self.memory.clear_memory()

    def max_lidar(self, observation, angle=np.pi / 3, pins=19):
        arg = np.argmax(observation)
        dir = -angle / 2 + arg * (angle / (pins - 1))
        dist = observation[arg]
        if arg == 0:
            distl = dist
        else:
            distl = observation[arg - 1]
        if arg == pins - 1:
            distr = dist
        else:
            distr = observation[arg + 1]
        return dir, (distl, dist, distr)

    def observe(self, racer_state):
        print("RACER_STATE = ", racer_state)
        if racer_state is None:
            return np.array([0])  # not used; we could return None
        else:
            lidar_signal, v = racer_state
            dir, (distl, dist, distr) = self.max_lidar(lidar_signal)
            return np.array([dir, distl, dist, distr, v])

    def fromObservationToModelState(self, observation):
        state = self.observe(observation)
        print("OBSERVATION = ", observation)
        print("STATE BEFORE EXPAND = ", state)
        state = tf.expand_dims(state, 0)
        print("STATE AFTER EXPAND = ", observation)
        return state

    # def logprobabilities(self, logits, a):
    #     num_actions = 2
    #     # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    #     logprobabilities_all = tf.nn.log_softmax(logits)
    #     logprobability = tf.reduce_sum(
    #         tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    #     )
    #     return logprobability

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
            self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, clip_ratio = 0.2
    ):

        # with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        #     ratio = tf.exp(
        #         self.logprobabilities(self.actor(observation_buffer), action_buffer)
        #         - logprobability_buffer
        #     )
        #     min_advantage = tf.where(
        #         advantage_buffer > 0,
        #         (1 + clip_ratio) * advantage_buffer,
        #         (1 - clip_ratio) * advantage_buffer,
        #     )
        #
        #     policy_loss = -tf.reduce_mean(
        #         tf.minimum(ratio * advantage_buffer, min_advantage)
        #     )
        # policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        # self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))
        #
        # kl = tf.reduce_mean(
        #     logprobability_buffer
        #     - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        # )
        # kl = tf.reduce_sum(kl)
        kl = 0.020
        return kl

    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def training2(self, state, racer, n_epochs=2, steps_per_epoch=5, gamma=0.99, alpha=0.0003,
                  gae_lambda=0.95,
                  policy_clip=0.2, train_policy_iterations = 3, train_value_iterations = 3, target_kl = 0.01):

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

                print("action= ", action)
                state_new, reward, done = racer.step(action)

                print("state_new", state_new)
                print("reward", reward)
                print("done", done)

                episode_return += reward
                episode_length += 1

                v_value = self.critic(state)


                # in teoria basta state, action, rewprd, value_t, logp_t per il training
                self.memory.store_memory(state_actual, action, reward, action_log_prob, v_value, done)

                self.memory.summary()

                # Update the state
                state_actual = state_new
                state_actual = self.fromObservationToModelState(state_actual)

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == steps_per_epoch - 1):
                    print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
                    last_value = 0 if done else self.critic(state_actual)
                    self.memory.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    print("NUM_EPISODES INCREMENTED: ", num_episodes)
                    state_actual, episode_return, episode_length = racer.reset(), 0, 0
                    state_actual = self.fromObservationToModelState(state_actual)

            # Get values from the buffer
            (
                states_buffer,
                actions_buffer,
                advantages_buffer,
                rewards_buffer,
                logprobabilities_buffer,
            ) = self.memory.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(train_policy_iterations):
                kl = self.train_policy(
                    states_buffer, actions_buffer, logprobabilities_buffer, advantages_buffer
                )
                if kl > 1.5 * target_kl:
                    # Early Stopping
                    print("EARLY STOPPING: KL= ", kl)
                    break

            # Update the value function
            for _ in range(train_value_iterations):
                print("TRAINING VALUE FUNCTION")
              # self.train_value_function(states_buffer, rewards_buffer)

            # Print mean return and length for each epoch
            print(" Epoch: ",ep + 1, ". Mean Return: ", sum_return / num_episodes, ". Mean Length: ", sum_length / num_episodes)

    def summary(self):
        print("The Agent is now working:\n")
        print("memory\n")
        self.memory.summary()
        # ..other information will be added later
        # self.actor.summary()
