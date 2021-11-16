import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import tracks

racer = tracks.Racer()

########################################
# Hyperparameters
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

num_states = 5  # we reduce the state dim through observation (see below)
num_actions = 2  # acceleration and steering
print("State Space dim: {}, Action Space dim: {}".format(num_states, num_actions))

upper_bound = 1
lower_bound = -1

print("Min and Max Value of Action: {}".format(lower_bound, upper_bound))


# The actor choose the move, given the state
def get_actor():

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(num_actions, name="out", activation='tanh')(out)

    model = tf.keras.Model(inputs, outputs, name="actor")
    return model


################
# observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
# logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
# actor = keras.Model(inputs=observation_input, outputs=logits)
#
# value = tf.squeeze(
#     mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
# )
# critic = keras.Model(inputs=observation_input, outputs=value)
#
# def mlp(x, sizes, activation=tf.tanh, output_activation=None):
#     # Build a feedforward neural network
#     for size in sizes[:-1]:
#         x = layers.Dense(units=size, activation=activation)(x)
#     return layers.Dense(units=sizes[-1], activation=output_activation)(x) #SIZES[-1???]
#################

def get_logits():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="tanh")(inputs)
    out = layers.Dense(64, activation="tanh")(out)
    # outputs = layers.Dense(num_actions, kernel_regularizer=regularizers.l2(0.01), kernel_initializer=last_init)(out)
    # outputs = layers.Activation('tanh')(outputs)
    # outputs = layers.Dense(num_actions, name="out", activation="tanh", kernel_initializer=last_init)(out)
    outputs = layers.Dense(num_actions, name="out", activation=None)(out)

    return outputs

def get_logits(train_acceleration=True,train_direction=True):

    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32, activation="tanh", trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32, activation="tanh", trainable=train_acceleration)(out1)
    out1 = layers.Dense(1, activation=None, trainable=train_acceleration)(out1)

    out2 = layers.Dense(32, activation="tanh", trainable=train_direction)(inputs)
    out2 = layers.Dense(32, activation="tanh", trainable=train_direction)(out2)
    out2 = layers.Dense(1, activation=None, trainable=train_direction)(out2)

    outputs = layers.concatenate([out1, out2])

    return outputs

def get_critic():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)  # Outputs single value for give state-action

    outputs = tf.squeeze(outputs, axis=1)

    model = tf.keras.Model(inputs, outputs, name="critic")

    return model


# Trajectories buffer
class Buffer:
    def __init__(self, observation_dimensions, size, gamma=0.99, lambd=0.95):
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def record(self, observation, action, reward, value, logprobability):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def discounted_cumulative_sums(x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def trajectory_advantage_rewards(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    print("observation: ", observation)
    logits = actor(observation)
    print("LOGITS: ", logits)
    randomElement= tf.random.categorical(logits, 2)
    print("randomElement: ", randomElement)
    action = tf.squeeze(randomElement)
    print("action: ", action)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


### CREATE MODELS ###

buffer = Buffer(num_states, steps_per_epoch)


actor = get_actor()
critic = get_critic()

actor.summary()
critic.summary()


# Initialize the policy and the value function optimizers
policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_function_learning_rate)

## TRAINING - DA RIGUARDARE!!!! ##


# we extract from the lidar signal the angle dir corresponding to maximal distance max_dir from track borders
# as well as the the distance at adjacent positions.
def max_lidar(observation, angle=np.pi / 3, pins=19):
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
    return (dir, (distl, dist, distr))


# input: state (actual observation) ; return dir (max distance from borders) + adjent to dir (distl, distr, dist)
# + v from actual observation
def observe(racer_state):
    if racer_state == None:
        return np.array([0])  # not used; we could return None
    else:
        lidar_signal, v = racer_state
        dir, (distl, dist, distr) = max_lidar(lidar_signal)
        return np.array([dir, distl, dist, distr, v])


observation, episode_return, episode_length = racer.reset(), 0, 0


def train():

    for ep in range(epochs):

        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        for t in range(steps_per_epoch):

            # Get the logits, action, and take one step in the environment
            observation2 = observe(observation)

            logits, action = sample_action(tf.expand_dims(tf.convert_to_tensor(observation2), 0))
            state_new, reward, done, _ = racer.step(action)

            print("state_new", state_new)
            print("reward", reward)
            print("done", done)

            episode_return += reward
            episode_length += 1


            # Get the value and log-probability of the action
            value_t = critic(state)
            logprobability_t = logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(state, action, reward, value_t, logprobability_t)

            # Update the state
            state = state_new


            # Finish trajectory if reached to a terminal state

            # we distinguish between termination with failure (state = None) and succesfull termination on track completion
            # succesfull termination is stored as a normal tuple
            fail = done and state == None

            buffer.record((prev_state, action, reward, fail, state))
            if not (done):
                mean_speed += state[4]
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observe(state))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                state, episode_return, episode_length = env.reset(), 0, 0

            # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
            )
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer)

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )


        ### TRAIN DDPG ###

        while not (done):
            i = i + 1

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            # our policy is always noisy
            action = policy(tf_prev_state)[0]
            # Get state and reward from the environment
            state, reward, done = racer.step(action)
            # we distinguish between termination with failure (state = None) and succesfull termination on track completion
            # succesfull termination is stored as a normal tuple
            fail = done and state == None
            state = observe(state)
            buffer.record((prev_state, action, reward, fail, state))
            if not (done):
                mean_speed += state[4]

            buffer.record((prev_state, action, reward, done, state))
            episodic_reward += reward

            states, actions, rewards, dones, newstates = buffer.sample_batch()
            targetQ = rewards + (1 - dones) * gamma * (target_critic([newstates, target_actor(newstates)]))

            loss1 = critic_model.train_on_batch([states, actions], targetQ)
            loss2 = aux_model.train_on_batch(states)

            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode {}: Avg. Reward = {}, Last reward = {}. Avg. speed = {}".format(ep, avg_reward, episodic_reward,
                                                                                       mean_speed / i))

        avg_reward_list.append(avg_reward)

    if total_episodes > 0:
        if save_weights:
            critic_model.save_weights("weights/ddpg_critic_weigths_32_car3_split.h5")
            actor_model.save_weights("weights/ddpg_actor_weigths_32_car3_split.h5")
        # Plotting Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Episodic Reward")
        plt.show()


train()

def actor(state):
    print("speed = {}".format(state[1]))
    state = observe(state)
    state = tf.expand_dims(state, 0)
    action = actor_model(state)
    print("acc = ", action[0, 0].numpy())
    return (action[0])


tracks.newrun(racer, actor)
