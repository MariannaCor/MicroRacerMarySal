import numpy as np
import tensorflow as tf
import tracks
from ppo.Agent2 import Agent2
from tensorflow import keras
from tensorflow.keras import layers

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

def new_race():
    print("\n\nNEW RACE ")


'''
Play game for n steps and store state, action probability, rewards, done variables.
Apply the Generalized Advantage Estimation method on the above experience. We will see this in the coding section.
Train neural networks for some epochs by calculating their respective loss.
Test this trained model for “m” episodes.
If the average reward of test episodes is larger than the target reward set by you then stop otherwise repeat from step one.
'''

def training_agent(agent, env):

    gamma = 0.99
    alpha = 0.0003
    gae_lambda = 0.95,
    policy_clip = 0.2,

    ##initialization
    n_epochs,steps_per_epoch = 2, 5
    train_policy_iterations, train_value_iterations = 3,3
    observation = env.reset()
    state = fromObservationToModelState(observation)

    for ep in range(n_epochs):
        sum_return = 0
        sum_length = 0
        num_episodes=0
        length_episodes= 0
        episode_return = 0
        episode_length= 0
        current_state = state

        for t in range(steps_per_epoch):

            action, dists = agent.act(current_state)
            print("dists ", dists)
            print("action ", action)
            observation, reward, done = env.step(action)
            v_value = agent.critic.model(current_state)

            print("observation ", observation)
            print("reward ", reward)
            print("done ", done)
            episode_return += reward
            episode_length += 1

            # in teoria bastano state, action_log_prob, reward, value_t
            agent.remember(current_state, action, dists, reward, v_value, done)
            #agent.summary()
            # model te new observation we got to the current_state
            current_state = fromObservationToModelState(observation)
            # Finish trajectory if reached to a terminal state
            terminal = done

            if terminal or (t == steps_per_epoch - 1):
                print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
                last_value = 0 if done else agent.critic.model(current_state)
                print("reward ====> ", reward)

                agent.calculate_advantages(last_value, gamma = 0.99,lam = 0.95)

                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                print("NUM_EPISODES INCREMENTED: ", num_episodes)
                observation, episode_return, episode_length = env.reset(), 0, 0
                current_state = fromObservationToModelState(observation)

        '''Train neural networks for some epochs by calculating their respective loss.'''
        agent.learn()
        agent.clean_memory()

        print(" Epoch: ",ep + 1, ". Mean Return: ", sum_return / num_episodes, ". Mean Length: ", sum_length / num_episodes)



if __name__ == '__main__':
    #tf.compat.v1.enable_eager_execution()
    #tf.executing_eagerly()
    print("tf.version = ", tf.version.VERSION)

    env = tracks.Racer()
    state_dim = 5  # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering

    agent = Agent2(state_dimension=state_dim, chunk_memory_size=10, num_actions = num_actions)

    doTrain = True
    doRace = True

    if doTrain:
        training_agent(agent, env)

    if doRace:
        new_race()
# def training2(self, state, racer, n_epochs=2, steps_per_epoch=5, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
#                   policy_clip=0.2, train_policy_iterations = 1, train_value_iterations = 1, target_kl = 0.01,
#                   ):
#         for ep in range(n_epochs):
#             # Initialize the sum of the returns, lengths and number of episodes for each epoch
#             sum_return = 0
#             sum_length = 0
#             num_episodes = 0
#             episode_return = 0
#             episode_length = 0
#             state_actual = state
#
#             for t in range(steps_per_epoch):
#
#                 action, action_log_prob = self.choose_action(state_actual)
#                 state_new, reward, done = racer.step(action)
#                 print("new_state ", state_new)
#                 print("reward ", reward)
#                 print("done ", done)
#
#                 episode_return += reward
#                 episode_length += 1
#
#                 v_value = self.critic.model(state_actual)
#
#                 # in teoria basta state, action, rewprd, value_t, logp_t per il training
#                 self.memory.store_memory(state_actual, action,  action_log_prob, reward, v_value, done)
#                 self.memory.summary()
#
#                 # Update the state
#                 state_actual = state_new
#                 state_actual = self.fromObservationToModelState(state_actual)
#
#                 # Finish trajectory if reached to a terminal state
#                 terminal = done
#
#                 if terminal or (t == steps_per_epoch - 1):
#                     print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
#                     last_value = 0 if done else self.critic.model(state_actual)
#                     print("reward ====> ",reward)
#                     self.memory.calculate_advantages(last_value)
#                     sum_return += episode_return
#                     sum_length += episode_length
#                     num_episodes += 1
#                     print("NUM_EPISODES INCREMENTED: ", num_episodes)
#                     state_actual, episode_return, episode_length = racer.reset(), 0, 0
#                     state_actual = self.fromObservationToModelState(state_actual)
