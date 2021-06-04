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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading

LOAD_FILE_EPISODE = 900  # Load Xth episode from file

MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

START_VIEW = 200

EPSILON = 0.0  # Epsilon

RENDER_GYM_WINDOW = False  # Opens a new window to render the game (Won't work on colab default)
RENDER_CV_WINDOW = True

# SALIENCY HYPERPARAMETERS
THRESHOLD=0.55
MODE='value' # 'value' or 'advantage', if 'advantage', parameter action=ACTION needs to be valid
ACTION=0
LAG=0 # WHICH FRAME YOU WANT TO GET SALIENCY FOR. 0 for most recent frame, -1 for average.
if __name__ == "__main__":
    environment = gym.make(ENVIRONMENT)  # Get env
    agent = Agent(environment)  # Create Agent
    agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))
    agent.online_model.eval()
    with open(MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get('epsilon')
        startEpisode = LOAD_FILE_EPISODE + 1
    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
    total_step = 1  # Cumulative sum of all steps in episodes
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
            ACTION=action
            environment.render()
            ataristate = agent.postProcess(state[0])

            img0=agent.getPosNegSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=0)
            #img0=agent.getAbsoluteSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=0)
            img1=agent.getPosNegSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=1)
            #img1=agent.getAbsoluteSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=1)
            img2=agent.getPosNegSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=2)
            #img2=agent.getAbsoluteSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=2)
            img3=agent.getPosNegSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=3)
            #img3=agent.getAbsoluteSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=3)
            img4=agent.getPosNegSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=-1)
            #img4=agent.getAbsoluteSaliencyImage(state,atariimg,mode=MODE,action=ACTION,threshold=THRESHOLD,lag=-1)
            if step > START_VIEW:
                # plt.imshow(img)
                # plt.show()
                cv2.imshow("Frame-0 (Last Frame)", img0)
                cv2.imshow("Frame-1", img1)
                cv2.imshow("Frame-2", img2)
                cv2.imshow("Frame-3", img3)
                cv2.imshow("Average Saliency", img4)
                cv2.waitKey(60)

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
