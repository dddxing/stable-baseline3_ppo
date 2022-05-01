import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from arm_gui import Renderer
from arm_dynamics import ArmDynamics
from robot import Robot
import time
import argparse

class ArmEnv(gym.Env):

    # ---------- IMPLEMENT YOUR ENVIRONMENT HERE ---------------------

    def __init__(self, arm):
        self.arm = arm  # DO NOT modify
        self.goal = None # Used for computing observation
        self.np_random = np.random # Use this for random numbers, as it will be seeded appropriately
        self.observation_space = None # You will need to set this appropriately
        self.action_space = None # You will need to set this appropriately
        # Fill in the rest of this function as needed

    # We will be calling this function to set the goal for your arm during testing.
    def set_goal(self,goal):
        self.goal = goal
        self.arm.goal = goal
        
    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # Fill in any additional functions you might need
