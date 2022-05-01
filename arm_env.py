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
    @staticmethod
    def cartesian_goal(radius,angle):
        return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1,1)

    @staticmethod
    def random_goal():
        radius_max = 2.0
        radius_min = 1.5
        angle_max = 0.5
        angle_min = -0.5
        radius = (radius_max - radius_min) * np.random.random_sample() + radius_min
        angle = (angle_max - angle_min) * np.random.random_sample() + angle_min
        angle -= np.pi/2
        return ArmEnv.cartesian_goal(radius, angle)

    def __init__(self, arm):
        self.arm = arm  # DO NOT modify
        self.goal = None # Used for computing observation
        self.np_random = np.random # Use this for random numbers, as it will be seeded appropriately
        
        # obs = q1, q2, qdot1, qdot2, xgoal, y_goal
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, ))

        a_space = arm.dynamics.get_action_dim()

        self.action_space = spaces.Box(-np.inf, np.inf, shape=(a_space, )) 
        # Fill in the rest of this function as needed
    
    # We will be calling this function to set the goal for your arm during testing.
    def set_goal(self,goal):
        self.goal = goal
        self.arm.goal = goal
        
    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal=None):
        self.arm.reset()
        if goal is None:
            self.goal = ArmEnv.random_goal()
        else:
            self.goal = goal
        self.arm.goal = self.goal
        self.num_steps = 0
        return np.append(self.arm.get_state(),self.goal).squeeze()

    def step(self, action):
        self.num_steps += 1

        #decoded_action = self._decode_action(action)
        self.arm.set_action(action)

        for _ in range(1):
            self.arm.advance()

        # if self.gui:
        #     self.renderer.plot([(self.arm, "tab:blue")])

        new_state = self.arm.get_state()

        # compute reward
        pos_ee = self.arm.dynamics.compute_fk(new_state)
        dist = np.linalg.norm(pos_ee - self.goal)
        vel_ee = np.linalg.norm(self.arm.dynamics.compute_vel_ee(new_state))
        reward = -dist**2 # change this if needed

        done = False
        if self.num_steps >= 500:
            done = True
        info = dict(pos_ee=pos_ee, vel_ee=vel_ee, success=True)
        observation = np.append(new_state,self.goal).squeeze()
        return observation, reward, done, info


    # Fill in any additional functions you might need
