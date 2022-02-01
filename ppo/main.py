import gc
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import animation

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

def new_race(env, agent, races=35):
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
    x = [*range(1, n_epoch+1)]
    plt.bar(x, line1,label=label)
    plt.xlabel('Epoch')
    plt.ylabel(' ')
    plt.title('Results')

    plt.show()


def training_agent(env,agent, n_epochs, steps_per_epoch, train_iteration, target_mean_diff):#races_for_test):
    print("Started Training ...\nHyperparameters: ")
    print(f"Actor_lr {agent.lr_actor}. Critic_lr {agent.lr_critic}. clip_value {agent.clip_ratio}. "
          f"N_epochs {n_epochs}. steps_per_epoch {steps_per_epoch}. train_iteraion {train_iteration}. " 
          f"target_mean_diff {target_mean_diff}. ")
          #f"Tests on {races_for_test} races")

    metric_a = []
    metric_b = []
    best_reward = float('-inf')
    best_avg_tested_reward = best_reward

    prev_reward = best_reward

    #params for advantage and return computation...

    # In situations when the agent should be acting in the present
    # in order to prepare for rewards in the distant future, this value should be large.
    # In cases when rewards are more immediate, it can be smaller.
    gamma = 0.99
    #Low values correspond to relying more on the current value estimate
    # (which can be high bias), and high values correspond to relying
    # more on the actual rewards received in the environment
    # (which can be high variance).
    # The parameter provides a trade-off between the two, and the
    # right value can lead to a more stable training process.
    lam = 0.97

    ##initialization
    episode_return, episode_length = 0,0

    observation = env.reset()
    state = fromObservationToModelState(observation)

    for ep in range(n_epochs):
        #print(ep+1, " EPOCH")
        gc.collect() #ogni cosa che non è puntanta viene cancellata
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        #print("Collecting new episodes")
        for t in range(steps_per_epoch):
            #if (t+1)% 1000 == 0:
                #print("collected {} steps ".format(t+1))

            #take a step into the environment
            action, dists = agent.act(state)
            observation, reward, done = env.step(action)
            #get the value of the critic
            v_value = agent.critic.model(state)
            agent.remember(state, action, dists, reward, v_value, done)
            episode_return += reward
            episode_length += 1
            #set the new state as current
            state = fromObservationToModelState(observation)
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


        agent.learn(training_iteration=train_iteration,target_mean_dif=target_mean_diff)

        # Print mean return and length for each epoch
        mean_episode_reward = sum_return / num_episodes
        metric_a.append(mean_episode_reward)
        metric_b.append(sum_length / num_episodes)

        #print( f" Epoch: {ep + 1}. Mean Return: {mean_episode_reward}. Trajectories: {num_episodes}. Mean Length: {sum_length / num_episodes} ")
        if(ep+1) % 10 == 0:
            print(f"saving models")
            agent.save_models(PATH_A)
            # # testing
            # steps, rewards = new_race(env, agent, races=races_for_test)
            # avg_tested_reward = tf.reduce_mean(rewards)
            # if (avg_tested_reward * 1000) >= (best_avg_tested_reward * 1000):
            #     best_avg_tested_reward = avg_tested_reward
            #     agent.save_models("best_tested")
            #     print("saved best_test !!")
            #
            # print(f"test : mean steps :  {tf.reduce_mean(steps)}. mean rewards : {avg_tested_reward}")


        toprint = "DOWN"
        if (mean_episode_reward * 1000) >= (prev_reward*1000):
            toprint = "UP  "
        prev_reward = mean_episode_reward

        xxx = "      "
        if (mean_episode_reward * 1000) >= (best_reward*1000):
            xxx = "Best !"
            best_reward = mean_episode_reward
            agent.save_models(PATH_B)

        toprint +=xxx
        print(f"Epoch: {ep+1}.\t{toprint} Mean Return: {mean_episode_reward}.\tTrajectories: {num_episodes}.\tMean Length: {sum_length / num_episodes}.")
        #print(f"Epoch: {ep}.\t{toprint} Collected Episodes: {num_episodes}. TEST: Reward: {mean_episode_reward}.\t Steps: {mean_steps}. ")

    agent.save_models(PATH_A)
    print("Training completed _\nMean Reward {}\nMean Length {}".format(metric_a,metric_b))
    plot_results(n_epochs, metric_a, "Mean Rewards")
    plot_results(n_epochs, metric_b, "Mean Length")

    return agent


def make_a_race(racer,agent):

    state = fromObservationToModelState(racer.reset())
    cs,csin,csout = racer.cs,racer.csin,racer.csout
    carx,cary = racer.carx,racer.cary
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = 2 * np.pi * np.linspace(0, 1, 200)
    ax.plot(csin(xs)[:, 0], csin(xs)[:, 1])
    ax.plot(csout(xs)[:, 0], csout(xs)[:, 1])
    ax.axes.set_aspect('equal')
    line, = plt.plot([], [], lw=2)
    xdata,ydata = [carx],[cary]

    acc = 0
    turn = 0

    def init():
        line.set_data([],[])
        return line,

    def counter():
        n = 0
        while not(racer.done):
            n += 1
            yield n

    def animate(i):
        nonlocal state
        action,_ = agent.act(state)
        a,d = np.squeeze(action)
        print(f"action {i}: acc {a}. turn {d}.")
        obs,reward,done = racer.step(action)
        state = fromObservationToModelState(obs)
        xdata.append(racer.carx)
        ydata.append(racer.cary)
        line.set_data(xdata, ydata)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=counter, interval=5, blit=True, repeat=False)
    plt.show()
#global variables
START_MODEL = "best_tested"
PATH_A = "saved_model"
PATH_B = "saved_best_model"


if __name__ == '__main__':
        tf.config.threading.set_intra_op_parallelism_threads(10)
        tf.config.threading.set_inter_op_parallelism_threads(10)
        # accumulator params for tests
        elapsed_time = 0
        steps, rewards = [], []
        #environment initialization
        env = tracks.Racer()
        # 1 is true, 0 is false
        doTrain = 0
        doRace  = 1
        doTest = 0

        #training params come in cartpole PPO keras : 30 epoche, 80 train_iteration e 40000 steps
        #training params in hands on 2049 steps size, PPO_EPOCHES = 10
        n_epochs = 100
        steps_per_epoch = 4096  #es: 1024, 2048, <3072>, 4096
        train_iteration = 50

        target_mean_diff = 0.0225# 1.5 * 0.015
        agent = Agent(
            load_models= True, path_saving_model= "saved_model",
            state_dimension = 5,
            num_action= 2,
            alpha = (3e-5, 1e-4), #lr_a,lr_c
            size_memory = steps_per_epoch,
            train_direction = True,
            train_acceleration = True,
            clip=.1
        )

        #race params
        number_of_races = 50

        if doTrain:
             agent = training_agent(
                 env, agent, n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, train_iteration=train_iteration,
                 target_mean_diff=target_mean_diff,# races_for_test=30
             )

        if doRace:
            agent.load_models("saved_model")
            for i in range(10):
                make_a_race(env,agent)

        if doTest:
            agent = Agent(
                load_models=True, path_saving_model="saved_model",
                state_dimension=5,
                num_action=2,
                alpha=(3e-5, 1e-4),  # lr_a,lr_c
                size_memory=steps_per_epoch,
                train_direction=True,
                train_acceleration=True,
                clip=.1
            )
            steps, rewards = new_race(env, agent, races=100)
            avg_tested_reward = tf.reduce_mean(rewards)
            print(f"test saved_model : mean steps :  {tf.reduce_mean(steps)}. mean rewards : {avg_tested_reward}")

            agent = Agent(
                load_models=True, path_saving_model="saved_best_model",
                state_dimension=5,
                num_action=2,
                alpha=(3e-5, 1e-4),  # lr_a,lr_c
                size_memory=steps_per_epoch,
                train_direction=True,
                train_acceleration=True,
                clip=.1
            )
            steps, rewards = new_race(env, agent, races=100)
            avg_tested_reward = tf.reduce_mean(rewards)
            print(f"test saved_best_model : mean steps :  {tf.reduce_mean(steps)}. mean rewards : {avg_tested_reward}")
