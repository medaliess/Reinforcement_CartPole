import os
import gym
import numpy as np

from collections import deque
import matplotlib.pyplot as plt
import pickle
# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Now, you can import TensorFlow
import tensorflow as tf
# Define the Policy Network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

# Define the REINFORCE Agent
class REINFORCEAgent:
    def __init__(self, state_size, action_size, gamma=0.9, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.policy_network = PolicyNetwork(state_size, action_size)

    def choose_action(self, state):
        state = np.array(state[0]) if isinstance(state, tuple) else np.array(state)
        probs = self.policy_network.predict(np.reshape(state, [1, self.state_size]))[0]
        action = np.random.choice(self.action_size, p=probs)
        return action

    def train(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)
            chosen_action_probs = tf.reduce_sum(tf.one_hot(actions, self.action_size) * action_probs, axis=1)
            loss = -tf.reduce_sum(tf.math.log(chosen_action_probs) * discounted_rewards)

        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

def run_reinforce(is_training=True, render=False, episodes=5):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)
    rewards_per_episode = []

    if not is_training:
        # Load saved weights
        with open('cartpole_reinforce_model_weights.pkl', 'rb') as f:
            weights = pickle.load(f)

        dummy_input = np.zeros((1, state_size))  # Use a dummy input with the appropriate shape
        _ = agent.policy_network(dummy_input)
        agent.policy_network.set_weights(weights)

    for episode in range(episodes):
        state = env.reset()
        terminated = False
        episode_states, episode_actions, episode_rewards = [], [], []

        while not terminated:
            action = agent.choose_action(state)
            next_state, reward, terminated, _ = env.step(action)[:4]

            state = np.array(state[0]) if isinstance(state, tuple) else np.array(state)
            next_state = np.array(next_state[0]) if isinstance(next_state, tuple) else np.array(next_state)
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
        if is_training:
            discounted_rewards = np.zeros_like(episode_rewards, dtype=np.float32)
            running_add = 0
            for t in reversed(range(len(episode_rewards))):
                running_add = running_add * agent.gamma + episode_rewards[t]
                discounted_rewards[t] = running_add

            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)

            states_tensor = tf.convert_to_tensor(np.array(episode_states), dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(np.array(episode_actions), dtype=tf.int32)
            discounted_rewards_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
            agent.train(states_tensor, actions_tensor, discounted_rewards_tensor)

        total_reward = sum(episode_rewards)
        rewards_per_episode.append(total_reward)
        print(f'Episode: {episode} | Total Reward: {total_reward}')

    env.close()

    if is_training:
        # Save target network weights using Pickle
        weights = agent.policy_network.get_weights()
        with open('cartpole_reinforce_model_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('REINFORCE Training on CartPole')
        plt.savefig('REINFORCE_cartpole_training_plot.png')

if __name__ == '__main__':
    run_reinforce(is_training=False, render=True, episodes=10)

