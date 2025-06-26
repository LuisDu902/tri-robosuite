# Evaluating Reward Structures for Robotic Arm Manipulation

**Intelligent Robotics Topics 2024/2025 | Masters in Artificial Intelligence**

Faculty of Engineering of University of Porto

## Group A1G

| Name           | Email                 |
| -------------- | --------------------- |
| Athos Freitas  | up202108792@fe.up.pt  |
| Luís Du       | up202105385@fe.up.pt  |
| Gonçalo Costa | up2002108814@fe.up.pt |

## Requirements


Ensure you have an environment with the following:

- Python 3.10+  
- [Mujoco](https://mujoco.readthedocs.io/) installed and properly licensed  
- [`robosuite`](https://github.com/ARISE-Initiative/robosuite) and its dependencies  
- PyTorch  
- Other required packages listed in `requirements.txt`

To install dependencies:

```bash
pip install -r requirements.txt
```

## How to run project

Clone the project:

```bash
git clone git@github.com:LuisDu902/tri-robosuite.git
cd tri-robosuite/robosuite
```

## Training Agents

Run training for a specific task:

```bash 
python -m rl_task.tasks.{task_name}.train
```

For example, to train the tuned stack task:

```bash
python -m rl_task.tasks.tuned-stack.train
```

## Monitoring Training

To visualize training metrics with TensorBoard:

```bash
tensorboard --logdir rl_task/tasks/{task_name}/logs
```

## Testing Agents

Run testing for a specific task:

```bash
python -m rl_task.tasks.{task_name}.test
```

For example, to train the tuned stack task:

```bash
python -m rl_task.tasks.tuned-stack.test
```
