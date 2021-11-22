import numpy as np
import tensorflow as tf
import tracks
from ppo.Agent2 import Agent2
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.enable_eager_execution()

#from MicroRacer_Corinaldesi_Fiorilla import tracks
#from MicroRacer_Corinaldesi_Fiorilla.ppo.Agent import Agent


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
    return dir, (distl, dist, distr)

def observe(racer_state):
    if racer_state == None:
        return np.array([0])  # not used; we could return None
    else:
        lidar_signal, v = racer_state
        dir, (distl, dist, distr) = max_lidar(lidar_signal)
        return np.array([dir, distl, dist, distr, v])


def fromObservationToModelState(observation):
    state = observe(observation)
    state = tf.expand_dims(state, 0)
    return state

def training2(self, state, racer, n_epochs=2, steps_per_epoch=5, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                  policy_clip=0.2, train_policy_iterations = 1, train_value_iterations = 1, target_kl = 0.01,
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


def new_race():
    pass


def training(agent, env):
    ##initialization
    n_epochs,steps_per_epoch = 2, 5
    train_policy_iterations, train_value_iterations = 1,1
    observation = env.reset()
    state = fromObservationToModelState(observation)

    for ep in range(n_epochs):
        sum_return = 0
        sum_length = 0
        num_episodes=0
        length_episodes=0
        current_state = state

        for t in range(steps_per_epoch):

            action, action_log_prob = agent.choose_action(current_state)
            v_value = agent.critic.model(current_state)
            new_observation, reward, done = env.step(action)

            print("new_observation ", new_observation)
            print("reward ", reward)
            print("done ", done)

            episode_return += reward
            episode_length += 1

            # in teoria basta state, action, action_log_prob, reward, value_t
            agent.remember(current_state, action, action_log_prob, reward, v_value, done)
            #agent.summary()

            # Update the state
            current_state = fromObservationToModelState(new_observation)

            # Finish trajectory if reached to a terminal state
            terminal = done

            if terminal or (t == steps_per_epoch - 1):
                print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
                last_value = 0 if done else agent.critic.model(current_state)
                print("reward ====> ", reward)

                agent.calculate_advantages(last_value)

                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                print("NUM_EPISODES INCREMENTED: ", num_episodes)
                state_actual, episode_return, episode_length = env.reset(), 0, 0
                state_actual = fromObservationToModelState(state_actual)


        print(" Epoch: ",ep + 1, ". Mean Return: ", sum_return / num_episodes, ". Mean Length: ", sum_length / num_episodes)



if __name__ == '__main__':

    print("tf.version = ", tf.version.VERSION)

    env = tracks.Racer()
    state_dim = 5  # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering

    agent = Agent2(state_dimension=state_dim, chunk_memory_size=10, num_actions = num_actions)


    doTrain = True
    doRace = True

    if doTrain:
        training(agent, env)

    if doRace:
        new_race()
