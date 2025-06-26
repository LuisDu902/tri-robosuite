import numpy as np

class HERReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions

        self.state_memory = np.zeros((max_size, *input_dims))
        self.new_state_memory = np.zeros((max_size, *input_dims))
        self.action_memory = np.zeros((max_size, n_actions))
        self.reward_memory = np.zeros(max_size)
        self.terminal_memory = np.zeros(max_size, dtype=bool)
        
        self.achieved_goals = np.zeros((max_size, 3))  # e.g., object position
        self.desired_goals = np.zeros((max_size, 3))

    def store_transition(self, state, action, reward, new_state, done, achieved_goal, desired_goal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.achieved_goals[index] = achieved_goal
        self.desired_goals[index] = desired_goal
        self.mem_cntr += 1

    def sample_buffer(self, batch_size, her_ratio=0.8):
        max_mem = min(self.mem_cntr, self.mem_size)
        indices = np.random.choice(max_mem, batch_size)

        states = self.state_memory[indices]
        next_states = self.new_state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices].copy()
        dones = self.terminal_memory[indices]
        ags = self.achieved_goals[indices]
        goals = self.desired_goals[indices]

        # HER relabeling
        for i in range(batch_size):
            if np.random.rand() < her_ratio:
                goals[i] = ags[i]  # Set goal to achieved goal
                # Recompute reward
                dist = np.linalg.norm(ags[i] - goals[i])
                rewards[i] = 1.0 if dist < 0.05 else 0.0  # Success condition

                # Repack state with new goal
                states[i][-3:] = goals[i]
                next_states[i][-3:] = goals[i]

        return states, actions, rewards, next_states, dones