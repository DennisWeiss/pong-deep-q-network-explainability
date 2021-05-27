import gym
import cv2
import matplotlib.pyplot as plt

import time
import json
import random
import numpy as np

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

START_VIEW=200


EPSILON = 0.0  # Epsilon

RENDER_GYM_WINDOW = False  # Opens a new window to render the game (Won't work on colab default)
RENDER_CV_WINDOW = True


if __name__ == "__main__":
    environment = gym.make(ENVIRONMENT)  # Get env
    agent = Agent(environment)  # Create Agent
    agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))
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
            environment.render()
            possaliency, negsaliency=agent.online_model.getSaliencyMap(state, 'advantage', action=action, PosNeg=True)
            possaliency=possaliency[0].cpu() #Use the saliency map for the most recent image
            negsaliency=negsaliency[0].cpu()
            possaliency+=state[0]#Add state to saliency maps in order to get gray game image
            negsaliency+=state[0]
            ataristate=agent.postProcess(state[0])
            ataripossaliency=agent.postProcess(possaliency.numpy())
            atarinegsaliency=agent.postProcess(negsaliency.numpy())
            img=torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0), torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0), torch.tensor(ataristate,dtype=torch.float).unsqueeze(0)])
            img=img/torch.max(img)
            img=img.transpose(0,2).transpose(0,1).numpy()
            img[0:20,:]=atariimg[0:20,:]/255.0
            if step>START_VIEW:
                #plt.imshow(img)
                #plt.show()
                img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("Display window", img)
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
