# IMPORTSsss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



# Define a single scalar Normal distribution.
#dist = tfd.Normal(loc=0., scale=3.)

class Memory:
     def __init__(self, chunk_size):
         self.states = []
         self.actions = []
         self.rewards = []
         self.values = []
         self.dones = []

         self.chunk_size = chunk_size

     def generate_batches(self):
         print("generates bathces ")
         n_states = len(self.states)
         #numpy.arrange([start, ]stop, [step, ], dtype = None) -> numpy.ndarray
         #es np.arange(0,10,2,float)  -> [0. 2. 4. 6. 8.]
         batch_start = np.arange(0, n_states, self.chunk_size)
         print( "batch_start = ", batch_start)
         indices = np.arange(n_states, dtype=np.int64)
         print("indices =", indices)
         np.random.shuffle(indices)
         batches = [indices[i:i + self.chunk_size] for i in batch_start]
         print("batches =", batches)

         return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.values), \
                np.array(self.rewards), \
                np.array(self.dones), \
                batches

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
        out1 = layers.Dense(256, activation="relu")(inputs)
        out1 = layers.Dense(256, activation="relu")(out1)
        # mu,var of accelleration
        mu_acc_out = layers.Dense(1, activation='tanh')(out1)
        var_acc_out = layers.Dense(1, activation='softplus')(out1)


        #direction
        out2 = layers.Dense(256, activation="relu")(inputs)
        out2 = layers.Dense(256, activation="relu")(out2)
        # mu,var of direction
        mu_dir_out = layers.Dense(1, activation='tanh')(out2)
        var_dir_out = layers.Dense(1, activation='softplus')(out1)

        outputs = layers.concatenate([mu_acc_out,var_acc_out,mu_dir_out,var_dir_out])

        self.model = tf.keras.Model(inputs, outputs, name="ActorNet")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



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


    def pdf(self, guess ,mean, sd):
        return (np.pi * sd) * np.exp(-0.5 * ((guess - mean) / sd) ** 2)


    def samplingAction(self,action):

        acc_dist,dir_dist = tf.split(value=action,num_or_size_splits=2, axis=1)

        print("acc_dist = ",acc_dist)
        print("dir_dist = ",dir_dist)

        mean_acc = acc_dist[0][0]
        stddev_acc = tf.sqrt(acc_dist[0][1])
        print("mean_acc = ",mean_acc)
        print("stddev_acc = ", stddev_acc)

        mean_dir = dir_dist[0][0]
        stddev_dir = tf.sqrt(dir_dist[0][1])
        print("mean_dir = ",mean_dir)
        print("stddev_dir = ", stddev_dir)

        print("questi li passo in input a random.normal ed ottengo dei numeri che squeezo")
        sampled_acc = tf.squeeze( tf.random.normal(shape=[], mean=mean_acc, stddev=stddev_acc) )
        sampled_dir = tf.squeeze( tf.random.normal( shape=[] ,mean= mean_dir , stddev=stddev_dir) )

        print("sampled_accelleration = ", sampled_acc)
        print("sampled_direction = ", sampled_dir)

        acc_prob = self.pdf(sampled_acc, mean=mean_acc, sd=stddev_acc)
        dir_prob = self.pdf(sampled_dir ,mean=mean_dir, sd=stddev_dir )

        print("acc_prob = ", acc_prob)
        print("dir_prob = ", dir_prob)

        #manca da creare le prob ed i valori in base ai range.. 

        #es di output: [ valore_accellerazione, valore_direzione ], [prob_accellerazione, prob_direzione]
        return sampled_actions,sampled_probs


    def choose_action(self, state):

        outputs = self.actor(state)
        v_value = self.critic(state)

        #poi campiono in base alla loro distribuzione.
        sampled_action,prob_action = self.samplingAction(outputs)

        return sampled_action,prob_action,v_value


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

    def training(self, n_epochs=50, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2):
        print("\n\n start learning new weights ..\n\n ")

        #definiamo un numero di epoche di cui vorremmo fare allenamento.
        for _ in range(n_epochs):
            #per ogni epoca generiamo una serie di osservazioni dalla self memory... la politica di generazione sta a noi definire come.
                state_arr, action_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
                #old_prob_arr,
                #mi sono salvato in memoria tutti i valori del critico man mano nell array vals.
                #mi creo per ogni epoca un array dove mi stimo l'advantage. questa sotto è l'inizializzazione.
                advantage = np.zeros(len(reward_arr), dtype=np.float32)
                #per ogni reward ottenuta in questo batch, mi calcolo l'advantage al tempo t, il discount per t+1 e
                #li metto dentro il mio array advantage man mano.

                for t in range(len(reward_arr) - 1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr) - 1):
                        a_t += discount * (reward_arr[k] + gamma * vals_arr[k + 1] * (1 - int(dones_arr[k])) - vals_arr[k])
                        discount *= gamma * gae_lambda
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






