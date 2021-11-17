import tracks
import tensorflow as tf
import numpy as np
from ppo.Agent import Agent


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

def fromObservationToModelState(observation):
    state = observe(observation)
    state = tf.expand_dims(state, 0)
    return state


if __name__ == '__main__':

    env = tracks.Racer()

    state_dim = 5   # we reduce the state dim through observation (see below)
    num_actions = 2  # acceleration and steering
    chunk_memory_size = 10

    agent = Agent(state_dimension = state_dim, num_actions=num_actions, chunk_memory_size = 10)
    agent.summary() #it prints a summary of the Agent

    done = False
    observation = env.reset() #prima osservazione
    state = fromObservationToModelState(observation)
    action = agent.choose_action(state)
    print("action squeezed is ", action)

    observation_, reward, done = env.step(action) # prima mossa

    agent.remember(observation, action, reward, done)
    agent.summary()
    #
    # n_games = 300
    # best_score = env.reward_range[0]
    # score_history = []
    #
    # learn_iters = 0 # mi serve per la print
    # avg_score = 0   #
    # n_steps = 0     #conto gli step per fermarmi ogni N
    #
    # for i in range(n_games):
    #     observation = env.reset()
    #     done = False
    #     score = 0
    #     while not done:
    #         action, prob, val = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         n_steps += 1
    #         score += reward
    #         agent.remember(observation, action, prob, val, reward, done)
    #         if n_steps % N == 0:
    #             agent.train()
    #             learn_iters += 1
    #         observation = observation_
    #     #score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])
    #
    #     if avg_score > best_score:
    #         best_score = avg_score
    #         agent.save_models()
    #
    #     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
    #             'time_steps', n_steps, 'learning_steps', learn_iters)
