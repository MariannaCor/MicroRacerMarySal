# IMPORTSsss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# class Memory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#
#         self.batch_size = batch_size
#
#     def generate_batches(self):
#         print("generating batches")
#
#     def store_memory(self):
#         print("storing memory..")
#
#     def clear_memory(self):
#         print("clearing memory")


class ActorNet():

    def __init__(self, input_dims, output_dims, lr=0.0003 ):
        #mancano da definire checkpoints e device di esecuzione

        # the actor has separate towers for action and speed
        # in this way we can train them separately

        inputs = layers.Input(shape=(input_dims,))
        #una torre stima l'accelleration
        out1 = layers.Dense(86, activation="relu")(inputs)
        out1 = layers.Dense(86, activation="relu")(out1)
        out1 = layers.Dense(1, activation='tanh')(out1)

        # una torre stima la direzione
        out2 = layers.Dense(86, activation="relu")(inputs)
        out2 = layers.Dense(86, activation="relu")(out2)
        out2 = layers.Dense(1, activation='tanh')(out2)

        outputs = layers.concatenate([out1, out2])

        # outputs = outputs * upper_bound #resize the range, if required
        self.model = tf.keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

      #  inputs = layers.Input(shape=(input_dims, ))
      #  out = layers.Dense(64, activation="relu")(inputs)
      #  out = layers.Dense(64, activation="relu")(out)
      #  outputs = layers.Dense(output_dims, name="out", activation="softmax")(out)

      #  self.model = tf.keras.Model(inputs, outputs, name="ActorNet")


    def getModel(self):
        return self.model

    def save_checkpoint(self): None
    def load_checkpoint(self): None


#
# class CriticNet():
#     def __init__(self):None
#     def save_checkpoint(self): None
#     def load_checkpoint(self): None


class Agent:

    def __init__(self,state_dimension, num_actions, alpha=0.0003 ):

        self.alpha = alpha
        self.state_dimension = state_dimension
        self.num_action = num_actions

        self.actor= ActorNet(input_dims= state_dimension , output_dims = self.num_action, lr=alpha ).getModel()

        #self.critic = CriticNet()
        #self.memory = Memory()

    def choose_action(self, state):
        print("Agent choose the action")
        print("input state is", state)
        mu_values = self.actor(state)
        action = tf.squeeze(mu_values)
        return action



    def remember(self, state, action, probs, vals, reward, done): None

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()



    def training(self):
        print("learning new weights ..")
        self.memory.clear_memory()



    def summary(self):
        print("The Agent is now working:\n")
        #..other information will be added later
        #self.actor.summary()




