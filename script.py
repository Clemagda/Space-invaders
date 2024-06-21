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
max_memory_length = 100000
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

# Fonction d'entraînement du modèle


def train_model(env, model, target_model, episodes=10, filename="training_output.mp4"):
    global frame_count, running_reward, episode_count, epsilon

    # Ajouter un writer pour enregistrer la vidéo de l'entraînement
    writer = imageio.get_writer(filename, fps=30)

    while True:
        # Reinitialise l'environnement et les récompenses à chaque épisode
        observation, _ = env.reset()
        state = np.array(observation)
        episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            # Via la probabilité Epsilon, le modèle choisit une action aléatoire.
            # Sinon, l'action avec la Q-value prédite la plus élevée est utilisée
            # TODO: se renseigner sur Epsilon greedy, probabilité epsilon, exploitation et exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                action = np.random.choice(num_actions)
            else:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

            # Ici, Epsilon est progressivement réduit jusqu'à un minimum
            # Le modèle prend de plus en plus d'actions éclairées
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            # Exécution de l'action et mise à jour de l'environnement
            state_next, reward, done, _, _ = env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Création d'un historique d'entraînement
            action_history.append(action)
            state_history.append(state)
            state_next_history.append(state_next)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
                indices = np.random.choice(
                    range(len(done_history)), size=batch_size)

                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array(
                    [state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices])

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

            # Mise à jour du modèle avec les poids du modèle principal
            if frame_count % update_target_network == 0:
                model_target.set_weights(model.get_weights())
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Suppression des anciennes transitions pour limiter la taille de l'historique
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break

            # Visualiser l'environnement à chaque étape
            frame = env.render()
            writer.append_data(frame)

        # Ajout de la récompense cumulée de l'épisode à l'historique et mise à jour
        # de la récompense moyenne
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        # Compte le nombre d'épisodes et arrête l'entraînement si l'épisode final est atteint ou que
        # la récompense moyenne actuelle atteint le palier de résolution
        episode_count += 1
        if running_reward >= 630:
            print(f"Solved at episode {episode_count}!")
            break

        if running_reward > 0 and episode_count >= max_episodes:
            print(f"Stopped at episode {episode_count}!")
            break

    writer.close()


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

train_model(env, model, model_target, episodes=50)
env.close()

# Evaluer les performances après entrainement et enregistrer la vidéo
env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode='rgb_array')
env = AtariPreprocessing(env)
env = FrameStack(4)
env.seed(seed)

scores_after = evaluate_and_record(
    env, model, episodes=5, filename="after_training.mp4")
env.close()
print(f"Scores after training : {scores_after}")
