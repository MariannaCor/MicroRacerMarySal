import numpy as np
import tensorflow as tf
#import tracks
import time
#from ppo.Agent2 import Agent2
from tensorflow import keras
from tensorflow.keras import layers

from MicroRacer_Corinaldesi_Fiorilla import tracks
from MicroRacer_Corinaldesi_Fiorilla.ppo.Agent2 import Agent2


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
            #print("state input =",state)
            action,prob = agent.act(state)
            #print("action =",action)
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
        for t in range(steps_per_epoch):
            #print("t step is "+str(t)+"in ep "+str(ep + 1))
            action, dists = agent.act(current_state)
            x = np.isnan(action)
            while x.any() : #ad un certo punto la rete torna valori nan ?a cosa è dovuto ?
                print("np.isnan (action ) =", x.any())
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

        '''Train neural networks for some epochs by calculating their respective loss.'''
        a_loss, c_loss = agent.learn(training_iteration=train_iteration)

        agent.clean_memory()

        #print(" A LOSS =",a_loss)
        #print(" C LOSS =", c_loss)

        #print(" Epoch: ",ep + 1, "Number of episodes :",num_episodes, " Average Reward ",np.mean(episodic_reward_list))

        if (ep+1) % 3 == 0:
            agent.save_models()

    return agent

if __name__ == '__main__':
    #tf.compat.v1.enable_eager_execution()
    #tf.executing_eagerly()
    print("tf.version = ", tf.version.VERSION)

    env = tracks.Racer()
    state_dim = 5  # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering
    alpha= 0.001 # learning rate


    # 1 is true, 0 is false
    doTrain = 0
    doRace = 1

    #accumulator params for tests
    elapsed_time = 0
    steps, rewards = [], []

    #training params come in cartpole: poche epoche, 80 train_iteration e 40000 steps
    n_epochs = 100
    steps_per_epoch = 100  ##quando da nan potrebbe essere
    train_iteration = 100

    ##race params
    number_of_races = 20

    agent = Agent2(state_dimension=state_dim, chunk_memory_size=steps_per_epoch, alpha=alpha)

    #if doRace:
    #    stepsA, rewardsA = new_race(env, agent, races=number_of_races)
   # steps, rewards = new_race(env, agent, races=number_of_races)
   # print_results(steps, rewards)

    if doTrain:
        t = time.process_time()
        # do some stuff
        agent = training_agent(env, agent,n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, train_iteration=train_iteration)
        elapsed_time = time.process_time() - t

    if doRace:
        #agent.load_models()
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
#
#
#
#
#           for batch in batches:
#                 states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
#                 old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
#                 actions = T.tensor(action_arr[batch]).to(self.actor.device)
#
#                 dist = self.actor(states)
#                 critic_value = self.critic(states)
#
#                 critic_value = T.squeeze(critic_value)
#
#                 new_probs = dist.log_prob(actions)

#                 #prob_ratio = (new_probs - old_probs).exp()
#                 weighted_probs = advantage[batch] * prob_ratio
#                 weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
#                         1+self.policy_clip)*advantage[batch]
#                 actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
#
#                 returns = advantage[batch] + values[batch]
#                 critic_loss = (returns-critic_value)**2
#                 critic_loss = critic_loss.mean()
#
#                 total_loss = actor_loss + 0.5*critic_loss
#                 self.actor.optimizer.zero_grad()
#                 self.critic.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.actor.optimizer.step()
#                 self.critic.optimizer.step()
