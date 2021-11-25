import numpy as np
import tensorflow as tf
import tracks
import time
from ppo.Agent2 import Agent2
from tensorflow import keras
from tensorflow.keras import layers

#from MicroRacer_Corinaldesi_Fiorilla import tracks
#from MicroRacer_Corinaldesi_Fiorilla.ppo.Agent2 import Agent2


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
    print("\n\nNEW RACES ")

    total_rewards = []
    total_steps = []

    for race in range(races):
        observation = env.reset()
        done = False
        steps_race_counter = 0
        rewards_for_race = []
        while not done:
            state = fromObservationToModelState(observation)
            print("state input =",state)
            action,prob = agent.act(state)
            print("action =",action)
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

'''

TODO:

- cancellare stampe non necessarie
- fare test di training:
    1. avviare n volte la corsa e salvare media rewards ottenute
    2. confrontare media
    3. tenere la migliore (salvare il modello)
    4. mostare parte grafica

'''

def training_agent(env,agent, n_epochs=20, steps_per_epoch=20, train_iteration=20):

    #for advantage computation
    gamma = 0.99
    lam = 0.95

    #for learning
    alpha = 0.0003 #learning rate
    policy_clip = 0.2

    ##initialization
    observation = env.reset()
    state = fromObservationToModelState(observation)

    for ep in range(n_epochs):

        num_episodes=0
        current_state = state

        for t in range(steps_per_epoch):

            action, dists = agent.act(current_state)
            v_value = agent.critic.model(current_state)
            observation, reward, done = env.step(action)

            agent.remember(current_state, action, dists, reward, v_value, done)

            #set the new state as current
            current_state = fromObservationToModelState(observation)

            #set terminal condition
            terminal = done

            # if The trajectory reached to a terminal state or the expected number we stop moving and we calculate advantage
            if terminal or (t == steps_per_epoch - 1):
                #print("DONE = ", terminal, " t == steps_per_epoch? ", (t == steps_per_epoch - 1))
                last_value = 0 if done else agent.critic.model(current_state)
                agent.calculate_advantages(last_value,gamma,lam)
                num_episodes += 1
                print("New Episode Starts. It's number: ", num_episodes)
                observation, episode_return, episode_length = env.reset(), 0, 0
                current_state = fromObservationToModelState(observation)

        '''Train neural networks for some epochs by calculating their respective loss.'''
        agent.learn(training_iteration=train_iteration)
        agent.clean_memory()
        print(" Epoch: ",ep + 1, "Number of episodes :",num_episodes)

    return agent

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

    #accumulator params for tests
    elapsed_time = 0
    steps, rewards = [], []

    #training params
    n_epochs = 200
    steps_per_epoch = 20
    train_iteration = 50

    ##race params
    number_of_races = 50

    if doRace:
        stepsA, rewardsA = new_race(env, agent, races=number_of_races)


    if doTrain:
        t = time.process_time()
        # do some stuff
        agent = training_agent(env, agent,n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, train_iteration=train_iteration)
        elapsed_time = time.process_time() - t

    if doRace:
        steps,rewards = new_race(env,agent,races=number_of_races)

    print("\nSummary of the " + str(number_of_races) + " races : \n")
    print("Total Reward => ", rewardsA)
    print("Steps done for race => ", stepsA)
    print("Mean Reward : ", np.mean(rewardsA))
    print("Mean Step Number : ", np.mean(stepsA))


    print("\nTest Completed\n\nTraining Summary\n")
    print("epoch number : " + str(n_epochs) + " steps_per_epoch " + str(steps_per_epoch) + " train_iteration " + str(train_iteration))
    print("Value in fractional seconds... Elapsed_training_time : ", elapsed_time)

    print("\nSummary of the " + str(number_of_races) + " races : \n")
    print("Total Reward => ", rewards)
    print("Steps done for race => ", steps)
    print("Mean Reward : ", np.mean(rewards))
    print("Mean Step Number : ", np.mean(steps))





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
