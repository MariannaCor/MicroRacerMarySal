import gc
import sys

import numpy as np
import tensorflow as tf

from utils import tracks
from forDebug.Agent_debug import Agent


def training_agent(env,agent, n_epochs, steps_per_epoch, train_iteration, target_kl):

    metric_a = []
    metric_b = []

    #params for advantage and return computation...
    gamma = 0.99
    lam = 0.95 #0.95 in hands on,0.995

    ##initialization
    episode_return, episode_length = 0,0

    observation = env.reset()
    state = fromObservationToModelState(observation)

    for ep in range(n_epochs):
        print(ep+1, " EPOCH")
        gc.collect()
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        print("Collecting new episodes")

        for t in range(steps_per_epoch):
            if (t+1)% 1000 == 0:
                print("collected {} episodes".format(t+1))

            #take a step into the environment
            action, dists = agent.act(state)

            if np.isnan(action).any() : #ad un certo punto la rete torna valori nan ?a cosa è dovuto ?
                sys.exit("np.isnan (action) = true")

            observation, reward, done = env.step(action)
            #get the value of the critic
            v_value = agent.critic.model(state)
            agent.remember(state, action, dists, reward, v_value, done)

            episode_return += reward
            episode_length += 1

            #set the new state as current
            state = fromObservationToModelState(observation)
            #set terminal condition

            # if The trajectory reached to a terminal state or the expected number we stop moving and we calculate advantage
            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic.model(state)

                agent.finish_trajectory(last_value, gamma, lam)

                #we reset env only when the episodes is over or the memory is full
                state = fromObservationToModelState(env.reset())
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                episode_return, episode_length = 0, 0

        #gc.collect()
        agent.learn(training_iteration=train_iteration,target_kl=target_kl)

        # Print mean return and length for each epoch
        metric_a.append(sum_return / num_episodes)
        metric_b.append(sum_length / num_episodes)
        print( f" Epoch: {ep + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}")

        if (ep+1) % 2 == 0:
           agent.save_models(PATH_B)

    print("Training completed _\nMean Reward {}\nMean Length {}".format(metric_a,metric_b))
    plot_results(n_epochs, metric_a, "Mean Return")
    plot_results(n_epochs, metric_b, "Mean Length")
    agent.save_models(PATH_B)
    return agent




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

PATH_A = "../saved_model"
PATH_B = "../saved_model"
if __name__ == '__main__':

    loadBeforeTraining = False

    learning_rates = 0.03,0.01

    n_epochs = 5
    steps_per_epoch = 10
    train_iteration = 100
    target_kl = 1.5 * 0.1
    pathA = PATH_A

    env = tracks.Racer()
    agent = Agent(
        load_models=loadBeforeTraining,
        path_saving_model=pathA,
        state_dimension=5,
        num_action=2,
        alpha=learning_rates,
        size_memory=steps_per_epoch
    )

    agent = training_agent(env, agent, n_epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                           train_iteration=train_iteration, target_kl=target_kl)
