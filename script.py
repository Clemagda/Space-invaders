# TODO: Réentrainer le modèle jusqu'au meilleur score
# Sauvegarder les poids du modèle


import random
from collections import deque
from datetime import datetime
from fileinput import filename
import os
from xml.sax.handler import feature_namespaces
import cv2
import imageio
import numpy as np
import tensorflow as tf
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import gymnasium as gym
from keras import layers
import keras

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Configuration des paramètres pour l'expérience
seed = 42
gamma = 0.99  # C'est le facteur de réduction pour les récompenses passées
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = (epsilon_max - epsilon_min)

batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10

num_actions = 6

# Helper function


def create_q_model():
    return keras.Sequential(
        [
            layers.Lambda(
                lambda tensor: tf.transpose(
                    tensor, [0, 2, 3, 1]),
                output_shape=(84, 84, 4),
                input_shape=(4, 84, 84),
            ),
            layers.Conv2D(32, 8, strides=4, activation='relu'),
            layers.Conv2D(64, 4, strides=2, activation='relu'),
            layers.Conv2D(64, 3, strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(num_actions, activation='linear'),
        ]
    )


# Créer le modèle Q
model = create_q_model()
model_target = create_q_model()
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_function = keras.losses.Huber()

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000000
update_after_actions = 4
update_target_network = 10000

# Fonction pour évaluer le modèle et enregistrer les vidéos


def evaluate_and_record(env, model, episodes=5, filename="output.mp4"):
    writer = imageio.get_writer(filename, fps=30)
    scores = []

    for episode in range(episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0

        while not done:
            state = np.expand_dims(state, axis=0)
            action_probs = model.predict(state)
            action = np.argmax(action_probs)

            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            total_reward += reward
            state = next_state

            # Render frame
            env.render()
            # Obtain the frame from the underlying environment
            frame = env.unwrapped.ale.getScreenRGB()
            writer.append_data(frame)

        scores.append(total_reward)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    writer.close()
    return scores

# Créer un échantillonage prioritaire


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, transition, error):
        priority = (error + 1e-5) ** self.alpha
        if not np.isfinite(priority):  # Vérifie si la priorité est finie
            priority = 1.0  # Utilise une valeur par défaut si elle n'est pas valide
        self.buffer.append(transition)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer")

        scaled_priorities = np.array(self.priorities) ** beta
        total_priority = sum(scaled_priorities)
        if not np.isfinite(total_priority) or total_priority == 0:
            raise ValueError("Total of weights must be finite and non-zero")

        sampling_probabilities = scaled_priorities / total_priority
        indices = random.choices(
            range(len(self.buffer)), k=batch_size, weights=sampling_probabilities)
        samples = [self.buffer[i] for i in indices]

        importance_sampling_weights = (
            len(self.buffer) * sampling_probabilities[indices]) ** (-beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        return samples, indices, importance_sampling_weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (error + 1e-5) ** self.alpha
            if not np.isfinite(priority):  # Vérifie si la priorité est finie
                priority = 1.0  # Utilise une valeur par défaut si elle n'est pas valide
            self.priorities[idx] = priority
# Fonction d'entraînement du modèle


def train_model(env, model, target_model, episodes=50, filename_prefix="training_output"):
    global frame_count, epsilon

    replay_buffer = PrioritizedReplayBuffer(capacity=max_memory_length)

    output_dir = "./Outputs/Training/"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%d%m%H%M")
    filename = f"{filename_prefix}_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)
    writer = imageio.get_writer(filepath, fps=30)

    for episode in range(episodes):
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            state_next, reward, done, _, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward

            target = reward + gamma * \
                np.max(model_target.predict(
                    np.expand_dims(state_next, axis=0))[0])
            current_q = np.max(model.predict(np.expand_dims(state, axis=0))[0])
            td_error = abs(target - current_q)

            replay_buffer.add(
                (state, action, reward, state_next, done), td_error)
            state = state_next

            if frame_count % update_after_actions == 0 and len(replay_buffer.buffer) > batch_size:
                try:
                    samples, indices, weights = replay_buffer.sample(
                        batch_size)
                except ValueError as e:
                    print(f"Error sampling from buffer: {e}")
                    continue

                state_sample, action_sample, rewards_sample, state_next_sample, done_sample = zip(
                    *samples)

                state_sample = np.array(state_sample)
                state_next_sample = np.array(state_next_sample)
                rewards_sample = np.array(rewards_sample)
                done_sample = np.array(done_sample)
                weights = np.array(weights)

                future_rewards = model_target.predict(state_next_sample)
                updated_q_values = rewards_sample + gamma * \
                    tf.reduce_max(future_rewards, axis=1)
                updated_q_values = updated_q_values * \
                    (1 - done_sample) - done_sample

                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    q_values = model(state_sample)
                    q_action = tf.reduce_sum(
                        tf.multiply(q_values, masks), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                td_errors = rewards_sample + gamma * \
                    tf.reduce_max(future_rewards, axis=1) - \
                    tf.reduce_max(model.predict(state_sample), axis=1)
                replay_buffer.update_priorities(indices, td_errors)

            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(episode_reward, episode, frame_count))

            if done:
                break

            frame = env.render()
            writer.append_data(frame)

        print(f"Episode {episode + 1} ended with reward {episode_reward}")
        if episode_reward >= 630:
            print(
                f"Solved at episode {episode + 1} with reward {episode_reward}!")
            break

    writer.close()

    weights_filename = f"{filename_prefix}_{timestamp}_weights.h5"
    weights_filepath = os.path.join(output_dir, weights_filename)
    model.save_weights(weights_filepath)
    print(f"Weights saved to {weights_filepath}")

    return model


# Évaluer les performances AVANT l'entraînement
env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode='human')
env = AtariPreprocessing(env)
env = FrameStack(env, 4)

scores_before = evaluate_and_record(
    env, model, episodes=5, filename="before_training.mp4")
env.close()
print(f"Scores before training: {scores_before}")
# Scores before training: [5.0, 55.0, 55.0, 5.0, 55.0]

# Entraîner le modèle
env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode='rgb_array')
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
env.seed(seed)

trained_model = train_model(env, model, model_target, episodes=20)
env.close()

# Evaluer les performances après entrainement et enregistrer la vidéo
env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode='rgb_array')
env = AtariPreprocessing(env)
env = FrameStack(4)
env.seed(seed)

scores_after = evaluate_and_record(
    env, model, episodes=10, filename="after_training_2406.mp4")
env.close()
print(f"Scores after training : {scores_after}")
# Scores after training: [100.0, 100.0, 40.0, 100.0, 40.0, 65.0, 130.0, 130.0, 40.0, 130.0]
