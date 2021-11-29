import sys
import time

import numpy as np
import tensorflow as tf
import gc

import tracks
from ppo.Agent2 import Agent2

#from MicroRacer_Corinaldesi_Fiorilla import tracks
#from MicroRacer_Corinaldesi_Fiorilla.ppo.Agent2 import Agent2
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


'''
Play game for n steps and store state, action probability, rewards, done variables.
Apply the Generalized Advantage Estimation method on the above experience. We will see this in the coding section.
Train neural networks for some epochs by calculating their respective loss.
Test this trained model for “m” episodes.
If the average reward of test episodes is larger than the target reward set by you then stop otherwise repeat from step one.
'''



def print_results(steps, rewards):
    # risultati delle corse dopo il training per ogni epoca
    print("Total Reward => ", rewards)
    print("Steps done for race => ", steps)
    print("Mean Reward : ", np.mean(rewards))
    print("Mean Step Number : ", np.mean(steps))
    print("###################################")


def training_agent(env,agent, n_epochs=20, steps_per_epoch=20, train_iteration=20):

    #for advantage computation
    gamma = 0.99
    lam = 0.97

    #for learning
    policy_clip = 0.2

    ##initialization
    observation = env.reset()
    state = fromObservationToModelState(observation)

    episodic_reward_list = []

    for ep in range(n_epochs):
        num_episodes=0
        current_state = state
        episodic_reward = 0
        print(ep+1, " EPOCH")
        gc.collect() #ogni cosa che non è puntanta viene cancellata
        agent.clean_memory() #pulisco la memoria.
        for t in range(steps_per_epoch):
            #print("t step is "+str(t)+"in ep "+str(ep + 1))
            action, dists = agent.act(current_state)

            if np.isnan(action).any() : #ad un certo punto la rete torna valori nan ?a cosa è dovuto ?
                print(x.any())
                sys.exit("np.isnan (action ) ")
                action, dists = agent.act(current_state)


            v_value = agent.critic.model(current_state)
            observation, reward, done = env.step(action)

            agent.remember(current_state, action, dists, reward, v_value, done)
            episodic_reward +=reward

            #set the new state as current
            current_state = fromObservationToModelState(observation)
            #set terminal condition
            terminal = done
            # if The trajectory reached to a terminal state or the expected number we stop moving and we calculate advantage
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic.model(current_state)
                agent.calculate_advantages(last_value,gamma,lam)
                num_episodes += 1
                #print("New Episode Starts. It's number: ", num_episodes)
                current_state = fromObservationToModelState(env.reset())
                episodic_reward_list.append(episodic_reward)
                episodic_reward=0

        agent.learn(training_iteration=train_iteration)

        agent.clean_memory()
        #print(" A LOSS =",a_loss)
        #print(" C LOSS =", c_loss)
        #print(" Epoch: ",ep + 1, "Number of episodes :",num_episodes, " Average Reward ",np.mean(episodic_reward_list))

        if (ep+1) % 10 == 0:
            agent.save_models(pathB)

    return agent

#global variables

pathA = "saved_model"
pathB = "saved_model"

if __name__ == '__main__':

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     #Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #          tf.config.set_logical_device_configuration(
    #              gpus[0],
    #              [tf.config.LogicalDeviceConfiguration(memory_limit=7680)])
    #          logical_gpus = tf.config.list_logical_devices('GPU')
    #          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #          # Virtual devices must be set before GPUs have been initialized
    #          print(e)


    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("tf.version = ", tf.version.VERSION)

    env = tracks.Racer()
    state_dim = 5  # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering

    alpha= 0.0001,0.0001 # learning rate: il primo è actor, il secondo è critic. ha senso tenerli diversi perchè a volte crasha una e non l'altra..


    # 1 is true, 0 is false
    doTrain = 0
    doRace = 1

    #accumulator params for tests
    elapsed_time = 0
    steps, rewards = [], []

    #training params come in cartpole: poche epoche, 80 train_iteration e 40000 steps
    n_epochs = 20
    steps_per_epoch = 250
    train_iteration = 100

    ##race params
    number_of_races = 50

    agent = Agent2(load_models=False,path_saving_model=pathA, state_dimension=state_dim,  alpha=alpha )

    #if doRace:
    # stepsA, rewardsA = new_race(env, agent, races=number_of_races)
    # steps, rewards = new_race(env, agent, races=number_of_races)
    # print_results(steps, rewards)

    if doTrain:
        t = time.process_time()
        agent = training_agent(env, agent, n_epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                               train_iteration=train_iteration)
        elapsed_time = time.process_time() - t

       # agent.summary(n_epochs)

        # do some stuff
        # try:
        #     # Specify an valid GPU device
        #     with tf.device('/GPU:0'):
        #
        # except RuntimeError as e:
        #     print(e)


    if doRace:
        agent.load_models(pathB)
        steps,rewards = new_race(env,agent,races=number_of_races)

    #print("\nSummary of the " + str(number_of_races) + " races : \n")
    #print("Total Reward => ", rewardsA)
    #print("Steps done for race => ", stepsA)
    #print("Mean Reward : ", np.mean(rewardsA))
    #print("Mean Step Number : ", np.mean(stepsA))

    print("\nTest Completed\n\nTraining Summary\n")
    print("epoch number : " + str(n_epochs) + " steps_per_epoch " + str(steps_per_epoch) + " train_iteration " + str(train_iteration))
    print("Value in fractional seconds... Elapsed_training_time : ", elapsed_time)

    print("\nSummary of the " + str(number_of_races) + " races : \n")
    print("Total Reward => ", rewards)
    print("Steps done for race => ", steps)
    print("Mean Reward : ", np.mean(rewards))
    print("Mean Step Number : ", np.mean(steps))

