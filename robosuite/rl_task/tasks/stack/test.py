import time
import os 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from rl_task.models import Agent
import torch as T

if __name__ == "__main__":
    if not os.path.exists("rl_task/tasks/stack/model"):
        os.makedirs("rl_task/tasks/stack/model")

    env_name = "Stack"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,              
        render_camera='frontview',
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        horizon=350,                   
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        checkpoint_dir="rl_task/tasks/stack/model"
    )

    # Check device
    if T.cuda.is_available():
        agent.device = T.device("cuda")
    elif T.backends.mps.is_available():
        agent.device = T.device("mps")
    else:
        agent.device = T.device("cpu")
    print("Testing on device:", agent.device)

    # Load the saved models
    agent.load_models()

    writer = SummaryWriter()

    # Fewer episodes for testing
    n_games = 50

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            # Use the greedy policy (no extra noise), so pass validation=True
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            score += reward

            # Optionally render every step
            env.render()

        writer.add_scalar("test_score", score, global_step=i)
        print(f"Test Episode {i}, Score {score}")