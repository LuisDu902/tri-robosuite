import time
import os 
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import CriticNetwork, ActorNetwork
from buffer import ReplayBuffer
from td3_torch import Agent
import torch as T

if __name__ == "__main__":

    # Create directory for saving models if it doesn't exist
    if not os.path.exists("rl_task/tasks/her/model"):
        os.makedirs("rl_task/tasks/her/model")

    # Switch to Stack environment
    env_name = "Stack"

    # Create the environment using robosuite, with a smaller horizon for faster training
    env = suite.make(
        env_name, 
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,         # <-- Keep headless for faster training
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,       # <-- Use reward shaping for better learning, gives reward based on distance to target
        control_freq=20,
        horizon=350,               # <-- Reduced horizon for quicker rollouts
    )

    # Wrap the environment to make it compatible with Gym
    env = GymWrapper(env)

    # Hyperparameters for TD3
    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    # Create an Agent
    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=(env.observation_space.shape[0] + 3,),  # Add 3 for the goal,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        checkpoint_dir="rl_task/tasks/her/model",
    )
    
    # Device selection: prefer CUDA -> MPS -> CPU
    print("Training on device:", agent.device)

    # Initialize TensorBoard writer
    writer = SummaryWriter("rl_task/tasks/her/logs")

    n_games = 1001

    best_score = -np.inf
    episode_identifier = (
        f"actor_lr={actor_learning_rate} critic_lr={critic_learning_rate} "
        f"batch_size={batch_size} layer1={layer1_size} layer2={layer2_size} "
        f"tau=0.005 env={env_name} {int(time.time())}"
    )

    # Optional: Load pretrained models if you have them
    #agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            # 1. Get current base cube (cubeB) position
            base_pos = env.env.sim.data.body_xpos[env.env.cubeB_body_id]

            # 2. Set dynamic stacking goal (above base cube)
            desired_goal = base_pos + np.array([0.0, 0.0, 0.05])

            # 3. Augment observation with goal for action selection
            obs_with_goal = np.concatenate((observation, desired_goal))

            # 4. Agent chooses an action based on the augmented observation
            action = agent.choose_action(obs_with_goal)

            # 5. Take action
            next_observation, reward, done, info = env.step(action)
            score += reward

            # 6. Get achieved goal (position of cubeA)
            achieved_goal = env.env.sim.data.body_xpos[env.env.cubeA_body_id]

            # 7. Augment next observation with same desired goal
            next_obs_with_goal = np.concatenate((next_observation, desired_goal))

            # 8. Store HER transition
            agent.remember(
                obs_with_goal, action, reward,
                next_obs_with_goal, done,
                achieved_goal, desired_goal
            )

            agent.learn()
            observation = next_observation
            
        # Log the score in TensorBoard
        writer.add_scalar(f"score/{episode_identifier}", score, global_step=i)

        # Save models every 10 episodes
        if i % 100 == 0:
            
            agent.save_models()
        
        print(f"Episode {i}, Score {score}")