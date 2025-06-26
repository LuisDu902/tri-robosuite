import numpy as np
from robosuite.environments.manipulation.stack import Stack

class CustomStack(Stack):
    """
    Custom Stack environment with separated lifting and aligning rewards.
    Inherits from the original Stack environment and overrides reward methods.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the custom stack environment."""
        super().__init__(*args, **kwargs)
    
    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:
                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Reaching reward: gripper close to cube A
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cubeA_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # Grasping reward
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if grasping_cubeA:
            r_reach += 0.25

        # Lifting reward: cube A above table by a margin
        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]
        cubeA_lifted = cubeA_height > table_height + 0.04
        r_lift = 1.0 if cubeA_lifted else 0.0

        # Alignment reward and penalty
        if cubeA_lifted:
            # Horizontal distance for alignment
            horiz_dist = np.linalg.norm(np.array(cubeA_pos[:2]) - np.array(cubeB_pos[:2]))
            r_align = 1.0 * (1 - np.tanh(5.0 * horiz_dist))  # Increased weight and adjusted scaling

            # Height difference penalty
            height_diff = abs(cubeA_height - cubeB_pos[2] - 0.05)  # Assume cubeB is at table height + cube size
            height_penalty = -0.5 * np.tanh(5.0 * height_diff)  # Penalize if too far vertically

            # Penalty for staying lifted without aligning
            alignment_penalty = -0.3 if horiz_dist > 0.06 else 0.0  # Penalty if horizontal distance is too large
            r_lift += r_align + height_penalty + alignment_penalty
        else:
            r_lift = 0.0

        # Stacking reward: cube A on cube B, not grasped
        r_stack = 0
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        if not grasping_cubeA and r_lift > 0 and cubeA_touching_cubeB:
            r_stack = 2.0

        return r_reach, r_lift, r_stack
    
    def get_reward_components(self):
        """
        Utility method to get individual reward components for logging/analysis.
        
        Returns:
            dict: Dictionary with individual reward components
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        return {
            'reach': r_reach,
            'lift': r_lift, 
            'stack': r_stack,
            'total': self.reward(None)
        }