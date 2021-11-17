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

        inputs = layers.Input(shape=(input_dims, ))
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(output_dims, name="out", activation="softmax")(out)

        self.model = tf.keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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

    def __init__(self,num_states, num_actions, alpha=0.0003 ):

        self.alpha = alpha
        self.num_states = num_states
        self.num_action = num_actions

        self.actor= ActorNet(input_dims= self.num_states, output_dims = self.num_action, lr=alpha).getModel()

        #self.critic = CriticNet()
        #self.memory = Memory()

    def choose_action(self, state):
        print("choose action called ")
        print("state is", state.numpy() )
        logits = self.actor(state)
        print("is returned ", logits.numpy())
        action = tf.squeeze(tf.random.categorical(logits, 2))

        return logits, action



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
        ##..other information will be added later
        ##self.actor.summary()




