# IMPORTSsss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Memory:
     def __init__(self, chunk_size):
         self.states = []
         self.actions = []
         self.rewards = []
         self.values = []
         self.dones = []

         self.size = chunk_size*4

     def generate_batches(self):
         print("Generating batches")

     def store_memory(self, state, action, reward,value, done):
         self.states.append(state)
         self.actions.append(action)
         self.rewards.append(reward)
         self.values.append(value)
         self.dones.append(done)


     def clear_memory(self):
         print("clearing memory")
         self.states = []
         self.actions = []
         self.rewards = []
         self.values = []
         self.dones = []

     def summary(self):
        sss = ""
        for i in range( len(self.states) ):
            sss += str(i)+" [\n"\
                "\t"+str(self.states[i][0])+ ",\n" \
                "\t" + str(self.states[i][1]) + ",\n" \
                "\t"+str(self.actions[i])+ ",\n"\
                "\t"+str(self.rewards[i]) + ",\n" \
                "\t" + str(self.values[i]) + ",\n" \
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
      #  outputs = layers.Dense(output_dims, name="out", activation="tanh")(out)
      #  self.model = tf.keras.Model(inputs, outputs, name="ActorNet")


    def getModel(self):
        return self.model

    def save_checkpoint(self): None
    def load_checkpoint(self): None


class CriticNet():
    def __init__(self, input_dims, lr=0.0003 ):

        # still missing to define checkpoints and device over execute the net: cpu vs gpu for instance.

        #input the state, output the value.
        inputs = layers.Input(shape=(input_dims,))
        out = layers.Dense(86, activation="relu" )(inputs)
        out = layers.Dense(64, activation="relu")(out)
        outputs = layers.Dense(1, activation ="relu")(out)

        self.model = tf.keras.Model(inputs, outputs, name="CriticNet" )
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)


    def save_checkpoint(self):None
    def load_checkpoint(self):None

    def getModel(self):
        return self.model


class Agent:

    def __init__(self,state_dimension, alpha=0.0003, chunk_memory_size=5):

        self.alpha = alpha
        self.state_dimension = state_dimension


        self.memory = Memory(chunk_memory_size)
        self.actor = ActorNet(input_dims= state_dimension, lr=alpha ).getModel()
        self.critic = CriticNet(input_dims=state_dimension, lr= alpha).getModel()


    def choose_action(self, state):
        mu_values = self.actor(state)
        action = tf.squeeze(mu_values)

        v_value = self.critic(state)

        return action,v_value



    def remember(self, state, action, reward, value, done):
        self.memory.store_memory(state,action,reward,value, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()



    def training(self):
        print("\n\n start learning new weights ..\n\n ")

        #definiamo un numero di epoche di cui vorremmo fare allenamento.
        for _ in range(self.n_epochs):
            #per ogni epoca generiamo una serie di osservazioni dalla self memory... la politica di generazione sta a noi definire come.
                state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

                #mi sono salvato in memoria tutti i valori del critico man mano nell array vals.
                #mi creo per ogni epoca un array dove mi stimo l'advantage. questa sotto è l'inizializzazione.
                advantage = np.zeros(len(reward_arr), dtype=np.float32)
                #per ogni reward ottenuta in questo batch, mi calcolo l'advantage al tempo t, il discount per t+1 e
                #li metto dentro il mio array advantage man mano.

                for t in range(len(reward_arr) - 1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr) - 1):
                        a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                        discount *= self.gamma * self.gae_lambda
                    advantage[t] = a_t
                # uscito dal ciclo converto values ed advantage corrispettivi in tensori tensorflow nell'esempio
                advantage = tf.convert_to_tensor(advantage)
                values = tf.convert_to_tensor(vals_arr)
                #ora che ho tutti i valori tranne uno per l'allenamento posso allenare... devo stimarmi le probabilità


                # for batch in batches:
                #     states = tf.convert_to_tensor(state_arr[batch], dtype="float64")
                #     old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                #     actions = tf.convert_to_tensor(action_arr[batch])
                #
                #     dist = self.actor(states)
                #     critic_value = self.critic(states)
                #     critic_value = tf.squeeze(critic_value)
                #
                #     new_probs = dist.log_prob(actions)
                #     prob_ratio = new_probs.exp() / old_probs.exp()
                #     # prob_ratio = (new_probs - old_probs).exp()
                #     weighted_probs = advantage[batch] * prob_ratio
                #     weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                #                                      1 + self.policy_clip) * advantage[batch]
                #
                #     actor_loss = -tf.minimum(weighted_probs, weighted_clipped_probs).mean()
                #
                #     returns = advantage[batch] + values[batch]
                #     critic_loss = (returns - critic_value) ** 2
                #     critic_loss = critic_loss.mean()
                #
                #     total_loss = actor_loss + 0.5 * critic_loss
                #     self.actor.optimizer.zero_grad()
                #     self.critic.optimizer.zero_grad()
                #     total_loss.backward()
                #     self.actor.optimizer.step()
                #     self.critic.optimizer.step()

        self.memory.clear_memory()



    def summary(self):
        print("The Agent is now working:\n")
        print("memory\n")
        self.memory.summary()
        #..other information will be added later
        #self.actor.summary()




