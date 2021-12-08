import gc
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import tracks
from ppo.Agent import Agent

# from MicroRacer_Corinaldesi_Fiorilla import tracks
# from MicroRacer_Corinaldesi_Fiorilla.ppo.Agent2 import Agent2
#tf.executing_eagerly()
#tf.compat.v1.enable_eager_execution()


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
def new_race(env, agent, races=15):

    #print("NEW RACES ")

    total_rewards = []
    total_steps = []

    for race in range(races):
        observation = env.reset()
        done = False
        steps_race_counter = 0
        rewards_for_race = []
        while not done:
            state = fromObservationToModelState(observation)
            action,prob = agent.act(state)
            observation, reward, done = env.step(action)
            steps_race_counter+=1
            rewards_for_race.append(reward)

        total_rewards.append(tf.reduce_sum(rewards_for_race).numpy())
        total_steps.append(steps_race_counter)

    return total_steps,total_rewards
def print_results(steps, rewards):
    # risultati delle corse dopo il training per ogni epoca
    print("Total Reward => ", rewards)
    print("Steps done for race => ", steps)
    print("Mean Reward : ", np.mean(rewards))
    print("Mean Step Number : ", np.mean(steps))
    print("###################################")

def plot_results(n_epoch, line1, label):
    #create array for x axis
    x = [*range(1, n_epoch+1)]
    plt.plot(x, line1, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(' ')
    plt.title('Results')
    plt.show()

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
        gc.collect() #ogni cosa che non è puntanta viene cancellata
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

        agent.learn(training_iteration=train_iteration,target_kl=target_kl)

        # Print mean return and length for each epoch
        metric_a.append(sum_return / num_episodes)
        metric_b.append(sum_length / num_episodes)
        print( f" Epoch: {ep + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}")

        if (ep+1) % 2 == 0:
           agent.save_models(pathB)

    print("Training completed _\nMean Reward {}\nMean Length {}".format(metric_a,metric_b))
    plot_results(n_epochs, metric_a, "Mean Return")
    plot_results(n_epochs, metric_b, "Mean Length")
    agent.save_models(pathB)
    return agent

#global variables
pathA = "saved_model"
pathB = "saved_model"


if __name__ == '__main__':

        # accumulator params for tests
        elapsed_time = 0
        steps, rewards = [], []

        #environment initialization
        env = tracks.Racer()

        # 1 is true, 0 is false
        doTrain = 1
        doRace = 1

        #training params come in cartpole PPO keras : 30 epoche, 80 train_iteration e 40000 steps
        #training params in hands on 2049 steps size, PPO_EPOCHES = 10
        n_epochs = 30
        steps_per_epoch = 2048  # es: 1024, 2048, 3072, 4096
        train_iteration = 100

        # lr_actor,lr_critic.
        learning_rates = 0.003, 0.001
        loadBeforeTraining = False
        target_kl = 1.5 * 0.1

        agent = Agent(
            load_models=loadBeforeTraining,
            path_saving_model = pathA,
            state_dimension = 5,
            num_action = 2 ,
            alpha = learning_rates,
            size_memory = steps_per_epoch
        )

        #race params
        number_of_races = 50
        # https://rishy.github.io/ml/2017/01/05/how-to-train-your-dnn/
        #
        # print("####### RACE BEFORE TRAIN #########")
        # steps_b, rewards_b = new_race(env, agent, races=number_of_races)
        # # ----PRINTING RESULTS-----------
        #
        # print("\nSummary of the " + str(number_of_races) + " races : \n")
        # print_results(steps_b, rewards_b)

        if doTrain:
             try:
                t = time.process_time()
                agent = training_agent(env, agent, n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, train_iteration=train_iteration, target_kl=target_kl)
                elapsed_time = time.process_time() - t
             except RuntimeError as e:
                print(e)

        if doRace:
            agent.load_models(pathB)
            steps,rewards = new_race(env,agent,races=number_of_races)

        #----PRINTING RESULTS-----------
        print("\nTest Completed\n\nTraining Summary\n")
        print("epoch number : " + str(n_epochs) + " steps_per_epoch " + str(steps_per_epoch) + " train_iteration " + str(train_iteration))

        print("Value in fractional seconds... Elapsed_training_time : ", elapsed_time)

        print("####### RACE AFTER TRAIN #########")
        print("\nSummary of the " + str(number_of_races) + " races : \n")
        print_results(steps, rewards)