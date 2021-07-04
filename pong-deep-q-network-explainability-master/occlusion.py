import gym
import cv2
import matplotlib.pyplot as plt

import time
import json
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from pong import Agent

ENVIRONMENT = "PongDeterministic-v4"

DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading

LOAD_FILE_EPISODE = 900  # Load Xth episode from file

MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

START_VIEW = 50

# OCCLUSION HYPERPARAMETERS
THRESHOLD=0.0
MODE='value' # 'value' , 'action' or 'advantage', if not 'value', parameter action=ACTION needs to be valid
# 'value' refers to the value estimation in the network, advantage stands for the advantage estimation and 'action' stands for the final logits
# The MODE determines what values will be used to compute the occlusion maps
ACTION=-1 # set -1 if you want saliency map for the whole action advantage vector/whole output vector of the network
CHOSENACTION=False # If this is true, ACTION will be updated each frame with the action that the agent chose last
CONCURRENT = False # If true, all regions are occluded at the same time in the 4 frames. If false, seperate maps for each frame is generated.
METHOD="Gaussian-Blur" # Currently "Box" or "Gaussian-Blur". If "Box" parameters Size, Stride and Color must be set
METRIC="Norm" # What value to compute from logits
SIZE=2.0
COLOR=0.28#233.0/3.0 # Grayscale value between 0 and 1 for the occlusion box color, if set to None, the average pixel value of the image will be used

if __name__ == "__main__":
    environment = gym.make(ENVIRONMENT)  # Get env
    agent = Agent(environment)  # Create Agent
    agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))
    #Put agent to evaluation mode (fix batch normalisation layer)
    agent.online_model.eval()


    with open(MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get('epsilon')
        startEpisode = LOAD_FILE_EPISODE + 1
    for episode in range(startEpisode, MAX_EPISODE):
        state = environment.reset()  # Reset env
        atariimg = state
        state = agent.preProcess(state)  # Process image

        # Stack state . Every state contains 4 consecutive frames
        # We stack frames like 4 channel image
        state = np.stack((state, state, state, state))

        for step in range(MAX_STEP):
            # Select and perform an action
            action = agent.act(state)  # Act
            if CHOSENACTION:
                ACTION=action
            environment.render()
            ataristate = agent.postProcess(state[0])
            if METHOD=="Box" or METHOD=="Gaussian-Blur":
                if step>START_VIEW:
                    img=agent.getOcclusionImage(state, method=METHOD, mode=MODE, action=ACTION, threshold=THRESHOLD, size=SIZE, color=COLOR, concurrent=CONCURRENT, metric=METRIC)
            else:
                raise ValueError("Invalid METHOD")
            if step > START_VIEW:
                cv2.imshow("Frame-0 (Last Frame)", cv2.resize(img, (400, 400)))
                cv2.imshow("atari image", atariimg)
                cv2.waitKey()

            next_state, reward, done, info = environment.step(action)  # Observe
            atariimg = next_state

            next_state = agent.preProcess(next_state)  # Process image

            # Stack state . Every state contains 4 time contionusly frames
            # We stack frames like 4 channel image
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            # Move to the next state
            state = next_state  # Update state

            if done:  # Episode completed
                break
