from numpy import sqrt
from numpy.random import randn
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

def skrand(skew, loc, scale):
    sigma = skew / sqrt(1.0 + skew ** 2) 

    (u0, v) = randn(2)
    u1 = sigma * u0 + sqrt(1.0 -sigma ** 2) * v 

    if u0 >= 0:
        return u1 * scale + loc 
    return (-u1) * scale + loc

class OverEstimationEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.reward_range = (-10, 10)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square (a distractor) in the center
        self.grid.set(width - 2, 1, Ball())

        # Place a large reward object in the corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = (1, height // 2)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Get the goal with the highest return."

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)
    
    def step(self, action):

        # Check if there is a ball in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        ball_ahead = front_cell and front_cell.type == 'ball'

        obs, reward, done, info = MiniGridEnv.step(self, action)

        # If the agent tries to walk over a ball
        if action == self.actions.forward and ball_ahead:
            reward = skrand(-6, 4, 4) - 0.9 * (self.step_count / self.max_steps)  
            done = True

        return obs, reward, done, info
    

class OverEstimationEnv5x5(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=5)

class OverEstimationRandomEnv5x5(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class OverEstimationEnv9x9(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=9)

class OverEstimationRandomEnv9x9(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=9, agent_start_pos=None)

class OverEstimationEnv11x11(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=11)

class OverEstimationRandomEnv11x11(OverEstimationEnv):
    def __init__(self):
        super().__init__(size=11, agent_start_pos=None)

register(
    id='MiniGrid-OverEstimation-5x5-v0',
    entry_point='gym_minigrid.envs:OverEstimationEnv5x5'
)

register(
    id='MiniGrid-OverEstimation-Random-5x5-v0',
    entry_point='gym_minigrid.envs:OverEstimationRandomEnv5x5'
)

register(
    id='MiniGrid-OverEstimation-9x9-v0',
    entry_point='gym_minigrid.envs:OverEstimationEnv9x9'
)

register(
    id='MiniGrid-OverEstimation-Random-9x9-v0',
    entry_point='gym_minigrid.envs:OverEstimationRandomEnv9x9'
)

register(
    id='MiniGrid-OverEstimation-11x11-v0',
    entry_point='gym_minigrid.envs:OverEstimationEnv11x11'
)

register(
    id='MiniGrid-OverEstimation-Random-11x11-v0',
    entry_point='gym_minigrid.envs:OverEstimationRandomEnv11x11'
)