import math

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

def Gaussian2DMatrix(mu,sigma, size):
    x,y=np.meshgrid()
def KLDivergence(p,q):
    return torch.dot(p,(torch.log(p)-torch.log(q)))

ENVIRONMENT = "PongDeterministic-v4"

DEVICE = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODELS = False  # Save models to file so you can test later
MODEL_PATH = "./models/pong-cnn-"  # Models path for saving or loading
SAVE_MODEL_INTERVAL = 10  # Save models at every X epoch
TRAIN_MODEL = False  # Train model while playing (Make it False when testing a model)

LOAD_MODEL_FROM_FILE = True  # Load model from file
LOAD_FILE_EPISODE = 900  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.99  # Epsilon decay rate by step

RENDER_GAME_WINDOW = True  # Opens a new window to render the game (Won't work on colab default)


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo. https://arxiv.org/abs/1511.06581
    """

    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.advantageEstimation = torch.empty(0, device=DEVICE, dtype=torch.float)
        self.valueEstimation = torch.empty(0, device=DEVICE, dtype=torch.float)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64  # Last conv layer's out sizes

        # Action layer
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer
        self.advantageEstimation = Ax.clone()

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer
        self.valueEstimation = Vx.clone()

        q = Vx + (Ax - Ax.mean())

        return q

    # Seperates the network computation into 4 graphs:
    # x->postConv1=self.preReLU1->self.postConv2=self.preReLU2->self.postConv3=self.preReLU3->y
    def guidedforward(self, x):
        self.postConv1 = self.bn1(self.conv1(x))
        self.preReLU1 = self.postConv1.detach().clone().requires_grad_(True)
        self.postConv2 = self.bn2(self.conv2(F.relu(self.preReLU1)))
        self.preReLU2 = self.postConv2.detach().clone().requires_grad_(True)
        self.postConv3 = self.bn3(self.conv3(F.relu(self.preReLU2)))
        self.preReLU3 = self.postConv3.detach().clone().requires_grad_(True)
        x = F.relu(self.preReLU3)

        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer
        self.advantageEstimation = Ax.clone()

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer
        self.valueEstimation = Vx.clone()

        q = Vx + (Ax - Ax.mean())

        return q

    # Inputs:
    # x: 4 channel input to the neural network
    #  mode= 'value' or 'advantage'
    #   If 'value':
    #       Return saliency map(s) associated with the value estimate
    #   If 'advantage':
    #       Then the input action has to be set to the index of the desired action to compute saliency map for
    # Returns 4xHxW dimensional 4 channel saliency map normalised in [-1,1] as a whole.

    def getGuidedBP(self, x, mode='value', action=None):
        if mode != 'value':
            if mode != 'advantage':
                raise ValueError("mode needs to be 'value' or 'advantage'!")
            else:
                if action == None or action < 0:
                    raise ValueError("If mode=='advantage', set non-negative action index to input 'action'.")
        self.zero_grad()
        inputs = torch.tensor(x, requires_grad=True, device=DEVICE, dtype=torch.float)
        self.guidedforward(inputs.unsqueeze(0))
        if mode == 'value':
            self.valueEstimation.backward()
        else:
            self.advantageEstimation[0][action].backward()
        self.postConv3.backward(gradient=F.threshold(self.preReLU3.grad, 0.0, 0.0))
        self.postConv2.backward(gradient=F.threshold(self.preReLU2.grad, 0.0, 0.0))
        self.postConv1.backward(gradient=F.threshold(self.preReLU1.grad, 0.0, 0.0))
        saliency = inputs.grad.clone()
        AbsSaliency = torch.abs(saliency.clone())
        saliency = saliency / torch.max(AbsSaliency)
        return saliency

    # Inputs:
    # x: 4 channel input to the neural network
    #  mode= 'value' or 'advantage'
    #   If 'value':
    #       Return saliency map(s) associated with the value estimate
    #   If 'advantage':
    #       Then the input action has to be set to the index of the desired action to compute saliency map for
    # Returns 4xHxW dimensional 4 channel saliency map normalised in [-1,1] as a whole.

    def getSaliencyMap(self, x, mode='value', action=None):
        if mode != 'value':
            if mode != 'advantage':
                raise ValueError("mode needs to be 'value' or 'advantage'!")
            else:
                if action == None or action < 0:
                    raise ValueError("If mode=='advantage', set non-negative action index to input 'action'.")
        self.zero_grad()
        inputs = torch.tensor(x, requires_grad=True, device=DEVICE, dtype=torch.float)
        self.forward(inputs.unsqueeze(0))
        if mode == 'value':
            self.valueEstimation.backward()
        else:
            self.advantageEstimation[0][action].backward()
        saliency = inputs.grad.clone()
        AbsSaliency = torch.abs(saliency.clone())
        saliency = saliency / torch.max(AbsSaliency)
        return saliency


class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition for Agent
        """
        # State size for breakout env. SS images (210, 160, 3). Used as input size in network
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Activation size for breakout env. Used as output size in network
        self.action_size = environment.action_space.n

        # Image pre process params
        self.original_h = 210
        self.original_w = 160
        self.target_h = 80  # Height after process
        self.target_w = 64  # Widht after process

        self.crop_dim = [20, self.state_size_h, 0,
                         self.state_size_w]  # Cut 20 px from top to get rid of the score table

        # Trust rate to our experiences
        self.gamma = GAMMA  # Discount coef for future predictions
        self.alpha = ALPHA  # Learning Rate

        # After many experinces epsilon will be 0.05
        # So we will do less Explore more Exploit
        self.epsilon = 1  # Explore or Exploit
        self.epsilon_decay = EPSILON_DECAY  # Adaptive Epsilon Decay Rate
        self.epsilon_minimum = 0.05  # Minimum for Explore

        # Deque holds replay mem.
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        # Create two model for DDQN algorithm
        self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # Adam used as optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image, singleChannel=False):
        """
        Process image crop resize, grayscale and normalize the images
        """
        #cv2.imshow("1", image)
        #cv2.waitKey()
        # plt.show()
        if not singleChannel:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
            # cv2.imshow("2",frame)
            # plt.show()
            frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
            # cv2.imshow("3",frame)
            # plt.show()
            frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
            # cv2.imshow("4", frame)
            # plt.show()
        else:
            frame=image

        frame = frame.reshape(self.target_w, self.target_h)
        if not singleChannel:
            frame = frame / 255.0 # Normalize
        # cv2.imshow("5",frame)
        # plt.show()
        # cv2.waitKey()
        return frame

    def postProcess(self, frame):
        """
        Undo image crop, resize and normalization of the images
        """
        # cv2.imshow("1-rec",frame)
        # plt.show()
        img = frame.reshape(self.target_h, self.target_w)  # * 255  # Normalize
        # cv2.imshow("2-rec", img)
        # plt.show()
        #img = cv2.resize(img, (self.original_w, self.original_h - 20))  # Resize
        # cv2.imshow("3-rec", img)
        # plt.show()
        #bigimg = np.zeros(shape=(self.original_h, self.original_w))
        # cv2.imshow("4-rec", bigimg)
        # plt.show()
        #bigimg[20:, :] = img
        # cv2.imshow("5-rec", bigimg)
        # plt.show()
        # cv2.waitKey()
        return img#bigimg

    def computeActivationDifference(self, state, occluded, mode='value', action=None, metric="KL"):
        if mode != 'value':
            if mode != 'advantage' and mode != 'action':
                raise ValueError("mode needs to be 'value', 'action' or 'advantage'!")
            else:
                if action == None or action < -1:
                    raise ValueError("If mode=='advantage' or 'action', set non-negative action index or -1 to input 'action'.")
        baseline=self.online_model.forward(torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0))
        if mode=='value':
            baseline=self.online_model.valueEstimation.detach().cpu()
        elif mode=='advantage':
            baseline=self.online_model.advantageEstimation.detach().cpu()

        occl=self.online_model.forward(torch.tensor(occluded, dtype=torch.float, device=DEVICE).unsqueeze(0))
        if mode=='value':
            occl=self.online_model.valueEstimation.detach().cpu()
        elif mode=='advantage':
            occl=self.online_model.advantageEstimation.detach().cpu()
        diff=baseline-occl
        if (mode=='advantage' or mode=='action') and action!=-1:
            diff=diff[action]
        if metric=="KL":
            return KLDivergence(F.softmax(baseline,dim=1).squeeze(0),F.softmax(occl,dim=1).squeeze(0))
        if metric=="JS":
            p=F.softmax(baseline,dim=1).squeeze(0)
            q=F.softmax(occl,dim=1).squeeze(0)
            r=(p+q)/2.0
            return 0.5*(KLDivergence(p,r)+KLDivergence(q,r))
        if metric=="Norm":
            return torch.linalg.norm(diff)

    # STRIDE IS NOT IMPLEMENTED. RIGHT NOW STRIDE=SIZE IS ASSUMED.
    def getBoxOcclusion(self, state, mode='value', action=None, size=3, stride=3, color=None, concurrent=False, metric="KL"):
        if color is None:
            color = np.mean(state, axis=(0, 1, 2))
        shape=self.postProcess(state[0]).shape
        imgs=np.zeros((4, shape[0], shape[1]))
        imgs[0]=self.postProcess(state[0])
        imgs[1]=self.postProcess(state[1])
        imgs[2]=self.postProcess(state[2])
        imgs[3]=self.postProcess(state[3])

        if not (self.preProcess(imgs[0], singleChannel=True)==state[0]).all():
            raise ValueError("Something went wrong")
        if not (self.preProcess(imgs[1], singleChannel=True)==state[1]).all():
            raise ValueError("Something went wrong")
        if not (self.preProcess(imgs[2], singleChannel=True)==state[2]).all():
            raise ValueError("Something went wrong")
        if not (self.preProcess(imgs[3], singleChannel=True)==state[3]).all():
            raise ValueError("Something went wrong")

        retimg=torch.zeros(imgs.shape) # Tensor the same shape as 4 frames of grayscale inputs.  If concurrent=True, all 4 maps are identical.
        for i in range(shape[0]):
            for j in range(shape[1]):
                box=np.zeros((4,shape[0],shape[1]))
                newstates=np.zeros(state.shape)
                if concurrent:
                    for k in range(4):
                        x_left = max(math.floor(i - (size / 2)), 0)
                        x_right = min(math.ceil(i + (size / 2)), shape[0])
                        y_top = max(math.floor(j - (size / 2)), 0)
                        y_bottom = min(math.ceil(j + (size / 2)), shape[1])
                        box[k, x_left:x_right, y_top:y_bottom] = np.ones((x_right - x_left, y_bottom - y_top)) * color
                    states=np.copy(imgs)
                    states[box>0]=box[box>0] # Occlusion
                    newstates[0]=self.preProcess(states[0], singleChannel=True)
                    newstates[1]=self.preProcess(states[1], singleChannel=True)
                    newstates[2]=self.preProcess(states[2], singleChannel=True)
                    newstates[3]=self.preProcess(states[3], singleChannel=True)
                    # COMPUTE ACTIVATION DIFFERENCE
                    sal=self.computeActivationDifference(state, newstates, mode=mode, action=action, metric=metric)
                    # RECORD SALIENCY
                    for k in range(4):
                        retimg[k,i,j] = sal
                else:
                    for k in range(4):
                        x_left = max(math.floor(i-(size/2)), 0)
                        x_right = min(math.ceil(i+(size/2)), shape[0])
                        y_top = max(math.floor(j-(size/2)), 0)
                        y_bottom = min(math.ceil(j+(size/2)), shape[1])
                        box[k, x_left:x_right, y_top:y_bottom] = np.ones((x_right - x_left, y_bottom - y_top)) * color
                        states=np.copy(imgs)
                        states[box>0]=box[box>0] # Occlusion
                        newstates[0] = self.preProcess(states[0], singleChannel=True)
                        newstates[1] = self.preProcess(states[1], singleChannel=True)
                        newstates[2] = self.preProcess(states[2], singleChannel=True)
                        newstates[3] = self.preProcess(states[3], singleChannel=True)
                        # COMPUTE ACTIVATION DIFFERENCE
                        sal=self.computeActivationDifference(state, newstates, mode=mode, action=action, metric=metric)
                        # RECORD SALIENCY
                        retimg[k,i,j] = sal
                        box = np.zeros((4, shape[0], shape[1]))
        return retimg


    def averageSaliencyMap(self, state, mode='value', action=None):
        saliency = self.online_model.getSaliencyMap(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = torch.sum(saliency, dim=0)
        saliency = saliency / 4
        return saliency

    def averageGuidedBP(self, state, mode='value', action=None):
        saliency = self.online_model.getGuidedBP(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = torch.sum(saliency, dim=0)
        saliency = saliency / 4
        return saliency

    def averageGaussianBlur(self, state, mode='value', action=None):
        saliency = self.online_model.getSaliencyMap(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = torch.sum(saliency, dim=0)
        saliency = saliency / 4
        return saliency

    def frameSaliencyMap(self, state, mode='value', action=None, lag=0):
        saliency = self.online_model.getSaliencyMap(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = saliency[lag]
        return saliency

    def frameGuidedBP(self, state, mode='value', action=None, lag=0):
        saliency = self.online_model.getGuidedBP(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = saliency[lag]
        return saliency

    def frameGaussianBlur(self, state, mode='value', action=None, lag=0):
        saliency = self.online_model.getGuidedBP(state, mode=mode, action=action)
        saliency = saliency.cpu()
        saliency = saliency[lag]
        return saliency

    def convertToPosNegSaliency(self, saliency):
        possaliency = F.threshold(saliency, 0.0, 0.0)
        negsaliency = F.threshold(-1 * saliency, 0.0, 0.0)
        return possaliency, negsaliency

    def convertToPositiveSaliency(self, saliency):
        possaliency = F.threshold(saliency, 0.0, 0.0)
        possaliency = possaliency / torch.max(possaliency)
        return possaliency

    def convertToNegativeSaliency(self, saliency):
        negsaliency = F.threshold(-1 * saliency, 0.0, 0.0)
        negsaliency = negsaliency / torch.max(negsaliency)
        return negsaliency

    def convertToAbsoluteSaliency(self, saliency):
        return torch.abs(saliency)

    def getAbsoluteSaliencyImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            saliency = self.convertToAbsoluteSaliency(self.averageSaliencyMap(state, mode=mode, action=action))
        else:
            saliency = self.convertToAbsoluteSaliency(self.frameSaliencyMap(state, mode=mode, action=action, lag=lag))
        ataristate = self.postProcess(state[0])
        saliency = saliency.cpu()
        if threshold > 0.0:
            saliency = F.threshold(saliency, threshold, 0.0)
        # saliency+=state[0]
        atarisaliency = self.postProcess(saliency.numpy())
        img = torch.cat([torch.tensor(atarisaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getAbsoluteGuidedBPImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            saliency = self.convertToAbsoluteSaliency(
                self.averageGuidedBP(state, mode=mode, action=action))
        else:
            saliency = self.convertToAbsoluteSaliency(
                self.frameGuidedBP(state, mode=mode, action=action, lag=lag))
        ataristate = self.postProcess(state[0])
        saliency = saliency.cpu()
        if threshold > 0.0:
            saliency = F.threshold(saliency, threshold, 0.0)
        # saliency += state[0]
        atarisaliency = self.postProcess(saliency.numpy())
        img = torch.cat([torch.tensor(atarisaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def getAbsoluteGaussianBlurImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            saliency = self.convertToAbsoluteSaliency(
                self.averageGuidedBP(state, mode=mode, action=action))
        else:
            saliency = self.convertToAbsoluteSaliency(
                self.frameGuidedBP(state, mode=mode, action=action, lag=lag))
        ataristate = self.postProcess(state[0])
        saliency = saliency.cpu()
        if threshold > 0.0:
            saliency = F.threshold(saliency, threshold, 0.0)
        # saliency += state[0]
        atarisaliency = self.postProcess(saliency.numpy())
        img = torch.cat([torch.tensor(atarisaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getPosNegSaliencyImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.averageSaliencyMap(state, mode=mode, action=action))
        else:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.frameSaliencyMap(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        #cv2.imshow("ataristate",ataristate)
        #cv2.waitKey()
        possaliency = possaliency.cpu()
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        ataripossaliency = self.postProcess(possaliency.numpy())
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.zeros(atarinegsaliency.shape, dtype=torch.float).unsqueeze(0),
                         torch.zeros(ataripossaliency.shape, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        # img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = ataristate/np.max(ataristate)
        return img

    def getPosNegGuidedBPImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.averageGuidedBP(state, mode=mode, action=action))
        else:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        possaliency = possaliency.cpu()
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        ataripossaliency = self.postProcess(possaliency.numpy())
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def getPosNegGaussianBlur(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.averageGuidedBP(state, mode=mode, action=action))
        else:
            possaliency, negsaliency = self.convertToPosNegSaliency(
                self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        possaliency = possaliency.cpu()
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        ataripossaliency = self.postProcess(possaliency.numpy())
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getPositiveSaliencyImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency = self.convertToPositiveSaliency(self.averageSaliencyMap(state, mode=mode, action=action))
        else:
            possaliency = self.convertToPositiveSaliency(
                self.frameSaliencyMap(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        possaliency = possaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        ataripossaliency = self.postProcess(possaliency.numpy())
        img = torch.cat([torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getPositiveGuidedBPImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency = self.convertToPositiveSaliency(self.averageGuidedBP(state, mode=mode, action=action))
        else:
            possaliency = self.convertToPositiveSaliency(self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        possaliency = possaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        ataripossaliency = self.postProcess(possaliency.numpy())
        img = torch.cat([torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def getPositiveGaussianBlurImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            possaliency = self.convertToPositiveSaliency(self.averageGuidedBP(state, mode=mode, action=action))
        else:
            possaliency = self.convertToPositiveSaliency(self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        possaliency = possaliency.cpu()
        if threshold > 0.0:
            possaliency = F.threshold(possaliency, threshold, 0.0)
        # possaliency += state[0]  # Add state to saliency maps in order to get gray game image
        ataripossaliency = self.postProcess(possaliency.numpy())
        img = torch.cat([torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataripossaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getNegativeSaliencyImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            negsaliency = self.convertToNegativeSaliency(self.averageSaliencyMap(state, mode=mode, action=action))
        else:
            negsaliency = self.convertToNegativeSaliency(
                self.frameSaliencyMap(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getNegativeGuidedBPImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            negsaliency = self.convertToNegativeSaliency(self.averageGuidedBP(state, mode=mode, action=action))
        else:
            negsaliency = self.convertToNegativeSaliency(self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getOcclusionImage(self, state, atariimg, method="Box", mode='value', action=None, threshold=0.0, size=3,
                          stride=3, color=None, concurrent=False, metric="KL"):

        ataristate = self.postProcess(state[0])

        if method == "Box":
            occmap = self.getBoxOcclusion(state, mode=mode, action=action, size=size, stride=stride, color=color,concurrent=concurrent, metric=metric)
        else:
            raise ValueError("Invalid method!")

        occmap = occmap.cpu()
        occmap /= torch.max(occmap)
        occmap = occmap ** (1/2)
        if threshold > 0.0:
            occmap = F.threshold(occmap, threshold, 0.0)

        # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        occlusion_maps=[]
        for i in range(4):
            map=torch.cat([occmap[i].detach().clone().unsqueeze(0),
                             occmap[i].detach().clone().unsqueeze(0),
                             torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
            #img = img / torch.max(img)
            map = map.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        #img[0:20, :] = atariimg[0:20, :] / 255.0
            occlusion_maps.append(cv2.cvtColor(map, cv2.COLOR_RGB2BGR))
        return occlusion_maps

    def getNegativeGaussianBlurImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1):
        if lag == -1:
            negsaliency = self.convertToNegativeSaliency(self.averageGuidedBP(state, mode=mode, action=action))
        else:
            negsaliency = self.convertToNegativeSaliency(self.frameGuidedBP(state, mode=mode, action=action, lag=lag))

        ataristate = self.postProcess(state[0])
        negsaliency = negsaliency.cpu()
        if threshold > 0.0:
            negsaliency = F.threshold(negsaliency, threshold, 0.0)
        # Add state to saliency maps in order to get gray game image
        # negsaliency += state[0]
        atarinegsaliency = self.postProcess(negsaliency.numpy())
        img = torch.cat([torch.tensor(atarinegsaliency, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0),
                         torch.tensor(ataristate, dtype=torch.float).unsqueeze(0)])
        img = img / torch.max(img)
        img = img.transpose(0, 2).transpose(0, 1).numpy()
        # print(np.max(img))
        img[0:20, :] = atariimg[0:20, :] / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def getSaliencyMapImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1, type='PosNeg'):
        if type != 'PosNeg' and type != 'Positive' and type != 'Negative' and type != 'Absolute':
            raise ValueError("type must be 'PosNeg', 'Positive', 'Negative' or 'Absolute'")
        elif type == 'PosNeg':
            img = self.getPosNegSaliencyImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Positive':
            img = self.getPositiveSaliencyImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Negative':
            img = self.getNegativeSaliencyImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Absolute':
            img = self.getAbsoluteSaliencyImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        return img

    def getGuidedBPImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1, type='PosNeg'):
        if type != 'PosNeg' and type != 'Positive' and type != 'Negative' and type != 'Absolute':
            raise ValueError("type must be 'PosNeg', 'Positive', 'Negative' or 'Absolute'")
        elif type == 'PosNeg':
            img = self.getPosNegGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Positive':
            img = self.getPositiveGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Negative':
            img = self.getNegativeGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Absolute':
            img = self.getAbsoluteGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        return img


    def getGaussianBlurImage(self, state, atariimg, mode='value', action=None, threshold=0.0, lag=-1, type='PosNeg'):
        if type != 'PosNeg' and type != 'Positive' and type != 'Negative' and type != 'Absolute':
            raise ValueError("type must be 'PosNeg', 'Positive', 'Negative' or 'Absolute'")
        elif type == 'PosNeg':
            img = self.getPosNegGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Positive':
            img = self.getPositiveGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Negative':
            img = self.getNegativeGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        elif type == 'Absolute':
            img = self.getAbsoluteGuidedBPImage(state, atariimg, mode=mode, action=action, threshold=threshold, lag=lag)
        return img

    def act(self, state):
        """
        Get state and do action
        Two option can be selectedd if explore select random action
        if exploit ask nnet for action
        """

        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

        return action

    def train(self):
        """
        Train neural nets with replay memory
        returns loss and max_q val predicted from online_net
        """
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        # We get out minibatch and turn it to numpy array
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        # Concat batches in one array
        # (np.arr, np.arr) ==> np.BIGarr
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # Find selected action's q_value
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(
            1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every step
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    environment = gym.make(ENVIRONMENT)  # Get env
    agent = Agent(environment)  # Create Agent

    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))
        with open(MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')

        startEpisode = LOAD_FILE_EPISODE + 1

    else:
        startEpisode = 1

    last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
    total_step = 1  # Cumulative sum of all steps in episodes
    for episode in range(startEpisode, MAX_EPISODE):

        startTime = time.time()  # Keep time
        state = environment.reset()  # Reset env

        state = agent.preProcess(state)  # Process image

        # Stack state . Every state contains 4 consecutive frames
        # We stack frames like 4 channel image
        state = np.stack((state, state, state, state))

        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode
        for step in range(MAX_STEP):
            # Select and perform an action
            action = agent.act(state)  # Act

            if RENDER_GAME_WINDOW:
                environment.render()  # Show state visually
                time.sleep(0.01)
                """for i in range(4):
                    plt.subplot(1,4,i+1)
                    gray=torch.tensor(state[0]).unsqueeze(0)
                    img=torch.cat([negsaliency[i].unsqueeze(0).cpu()+gray, possaliency[i].unsqueeze(0).cpu()+gray, gray])#torch.zeros(possaliency[i].unsqueeze(0).size())])
                    img=img/torch.max(img)
                    plt.imshow(gray[0])#.transpose(0,2).transpose(0,1))
                plt.show()
                exit()"""

            next_state, reward, done, info = environment.step(action)  # Observe

            next_state = agent.preProcess(next_state)  # Process image

            # Stack state . Every state contains 4 time contionusly frames
            # We stack frames like 4 channel image
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            # Store the transition in memory
            agent.storeResults(state, action, reward, next_state, done)  # Store to mem

            # Move to the next state
            state = next_state  # Update state

            if TRAIN_MODEL:
                # Perform one step of the optimization (on the target network)
                loss, max_q_val = agent.train()  # Train with random BATCH_SIZE state taken from mem
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_step += 1
            if total_step % 1000 == 0:
                agent.adaptiveEpsilon()  # Decrase epsilon

            if done:  # Episode completed
                currentTime = time.time()  # Keep current time
                time_passed = currentTime - startTime  # Find episode duration
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
                epsilonDict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file

                if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  # Save model as file
                    weightsPath = MODEL_PATH + str(episode) + '.pkl'
                    epsilonPath = MODEL_PATH + str(episode) + '.json'

                    torch.save(agent.online_model.state_dict(), weightsPath)
                    with open(epsilonPath, 'w') as outfile:
                        json.dump(epsilonDict, outfile)

                if TRAIN_MODEL:
                    agent.target_model.load_state_dict(agent.online_model.state_dict())  # Update target model

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
                    episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val,
                    agent.epsilon, time_passed, step, total_step
                )

                print(outStr)

                if SAVE_MODELS:
                    outputPath = MODEL_PATH + "out" + '.txt'  # Save outStr to file
                    with open(outputPath, 'a') as outfile:
                        outfile.write(outStr + "\n")

                break
