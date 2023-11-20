import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf



class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

class DDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, epsilon=.1, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.buffer = []
        self.batch_size = 32

        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())

        # Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def choose_action(self, state):
        state = np.array(state[0]) if isinstance(state, tuple) else np.array(state)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.q_network.predict(np.reshape(state, [1, self.state_size]))
            return np.argmax(q_values[0])
        


    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        
        print('training')

        while len(self.buffer)>=self.batch_size :
        
            batch = self.buffer[:self.batch_size]
            self.buffer=self.buffer[self.batch_size:]
            states = []
            targets = []

            for sample in batch:
                state, action, reward, next_state, done = sample
                state = np.array(state[0]) if isinstance(state, tuple) else np.array(state)

                next_state = np.array(next_state[0]) if isinstance(next_state, tuple) else np.array(next_state)
                target_next = self.target_network.predict(np.reshape(next_state, [1, self.state_size]))

                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(target_next[0])

                states.append(state)
                targets.append(target)

            states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
            targets_tensor = tf.convert_to_tensor(np.array(targets), dtype=tf.float32)

            # TensorFlow loss calculation
            with tf.GradientTape() as tape:
                q_values = self.q_network(states_tensor)
                selected_action_values = tf.gather(q_values, action, axis=1)
                losss = tf.reduce_mean(tf.square(targets_tensor - selected_action_values))
                

            gradients = tape.gradient(losss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
    
        # update target network weights each 10 episodes
        self.target_network.set_weights(self.q_network.get_weights())
        



def run_ddqn(is_training=True, render=False,episodes=5):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    rewards_per_episode = []

    if not is_training:
        # Load saved weights
        with open('cartpole_DDQN_model_weights.pkl', 'rb') as f:
            weights = pickle.load(f)

        dummy_input = np.zeros((1, state_size))  # Use a dummy input with the appropriate shape
        _ = agent.q_network(dummy_input)

        # Set the weights
        agent.q_network.set_weights(weights)
      

    for episode in range(episodes):  # You may adjust the number of episodes
        state = env.reset()
        terminated = False
        total_reward = 0

        while not terminated:
            action = agent.choose_action(state)
            next_state, reward, terminated, _ = env.step(action)[:4]

            agent.buffer.append([state, action, reward, next_state, terminated])
            total_reward += reward
            state = next_state

        rewards_per_episode.append(total_reward)
        print(f'Episode: {episode} | total reward: {total_reward}')   
        #train the agent each 10 episodes
        if is_training and episode%10==0:
            agent.train()
        
        
        
    env.close()

    if is_training:

        target_weights = agent.target_network.get_weights()
        with open('cartpole_DDQN_model_weights.pkl', 'wb') as f:
            pickle.dump(target_weights, f)

        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reinforce Training on CartPole')
        plt.savefig('DDQN_cartpole_training_plot.png')


if __name__ == '__main__':
    run_ddqn(is_training=False, render=True,episodes=10)