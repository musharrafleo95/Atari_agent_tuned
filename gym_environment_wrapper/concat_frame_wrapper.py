
from collections import deque

# RL imports
import gym

#image processing imports
import cv2

# other ml library
import numpy as np

# writing environment wrapper to concatenate frames to give RL algorithm temporal information

class ConcateObs(gym.Wrapper):
    """
    "Atari wrapper to create concatenated frames"

    env: gym environment
    k: the number of frames we want to concatenate
    seed: for reproduciblity
    """
    def __init__(self, env, k, seed = 0, normalize_frame = True, resized_x = 80, resized_y = 80):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        obs_shape = (resized_x, resized_y)

        if normalize_frame:
            high = 1
        else:
            high = 255

        self.observation_space = gym.spaces.Box(
            low=0,
            high=high,
            shape=((k,) + obs_shape),
            dtype=env.observation_space.dtype
        )

        self.seed_set = seed
        self.normalize_frame = normalize_frame
        self.obs_shape = obs_shape

    def _get_obs(self):
        return np.array(self.frames)

    def reset(self):
        obs = self.env.reset(seed=self.seed_set)
        # creating initial concatenated frame. since no previous frames available so combining same frames
        for _ in range(self.k):
            self.frames.append(self.frame_preprocessing(obs))
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(self.frame_preprocessing(obs))
        return self._get_obs(), reward, done, info

    def frame_preprocessing(self, obs):
        # converting image to gray
        gray_obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # resizing image
        rs_gray_obs = cv2.resize(gray_obs, self.obs_shape)
        # whether to normalize. However CnnPolicy do it automatically
        if self.normalize_frame:
            rs_gray_obs = rs_gray_obs/255.

        return rs_gray_obs
