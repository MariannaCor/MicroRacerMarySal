# IMPORTSsss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Memory:
     def __init__(self, chunk_size):
         self.states = []
         self.actions = []
         self.rewards = []
         self.dones = []

         self.size = chunk_size*4

     def generate_batches(self):
         print("Generating batches")

     def store_memory(self, state, action, reward, done):
         self.states.append(state)
         self.actions.append(action)
         self.rewards.append(reward)
         self.dones.append(done)
         print("storing memory..")

     def clear_memory(self):
         print("clearing memory")
         self.states = []
         self.actions = []
         self.rewards = []
         self.dones = []

     def summary(self):
        sss = ""
        for i in range(len( self.states ) ):
            sss += str(i)+" [\n"\
                "\t"+str(self.states[i][0])+ ",\n" \
                "\t" + str(self.states[i][1]) + ",\n" \
                "\t"+str(self.actions[i])+ ",\n"\
                "\t"+str(self.rewards[i]) + ",\n"\
                "\t"+str(self.dones[i]) + " ]\n\n"
        print(sss)

class ActorNet():

    def __init__(self, input_dims, lr=0.0003 ):
        #mancano da definire checkpoints e device di esecuzione

        # the actor has separate towers for direction and acceleration
        # in this way we can train them separately

        inputs = layers.Input(shape=(input_dims,))
        #acceleration
        out1 = layers.Dense(86, activation="relu")(inputs)
        out1 = layers.Dense(86, activation="relu")(out1)
        out1 = layers.Dense(1, activation='tanh')(out1)

        #direction
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

    def __init__(self,state_dimension, num_actions, alpha=0.0003, chunk_memory_size=5):

        self.alpha = alpha
        self.state_dimension = state_dimension
        self.num_action = num_actions

        self.memory = Memory(chunk_memory_size)
        self.actor= ActorNet(input_dims= state_dimension, lr=alpha ).getModel()
        #self.critic = CriticNet()


    def choose_action(self, state):
        print("Agent choose the action")
        print("input state is", state)
        mu_values = self.actor(state)
        action = tf.squeeze(mu_values)
        return action



    def remember(self, state, action, reward, done):
        self.memory.store_memory(state,action,reward,done)

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
        print("memory\n")
        self.memory.summary()
        #..other information will be added later
        #self.actor.summary()




