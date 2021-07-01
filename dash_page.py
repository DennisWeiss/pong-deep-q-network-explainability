import base64

import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from chart_studio import plotly
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gym
import keyboard
import json
import time
from collections import deque

import pong

ENVIRONMENT = "PongDeterministic-v4"
DEVICE = torch.device('cpu')
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

# OCCLUSIONB HYPERPARAMETERS
THRESHOLD=0.0
MODE='value' # 'value' , 'action' or 'advantage', if not 'value', parameter action=ACTION needs to be valid
# 'value' refers to the value estimation in the network, advantage stands for the advantage estimation and 'action' stands for the final logits
# The MODE determines what values will be used to compute the occlusion maps
ACTION=-1 # set -1 if you want saliency map for the whole action advantage vector/whole output vector of the network
CHOSENACTION=False # If this is true, ACTION will be updated each frame with the action that the agent chose last
CONCURRENT = False # If true, all regions are occluded at the same time in the 4 frames. If false, seperate maps for each frame is generated.
METRIC="Norm" # What value to compute from logits
SIZE=2.0

app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children=[
        html.Img(
            id='game-screen'
        ),
    ],
        style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
    html.Div(children=[
        html.Img(
            id='saliency-map'
        )
    ],
        style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
    html.Img(
        id='action-tree',
        width=1800,
    ),
    dcc.Interval(
        id='interval-component',
        interval=4000,
        n_intervals=0
    )
])

action_dict = {
    'NOOP': 'x',
    'FIRE': 'O',
    'LEFT': '<-',
    'RIGHT': '->',
    'LEFTFIRE': '<-O',
    'RIGHTFIRE': 'O->'
}

fig = None


def showActionTree(env, agent, state, episode, step, number_steps_ahead):
    global fig

    SCALE = 4
    Q_VALUE_SENSITIVITY = 30

    actions = env.unwrapped.get_action_meanings()
    snapshot = env.ale.cloneState()
    current_snapshot = None
    actionTree = nx.Graph()
    actionTree.add_node('0', pos=(0, 0))

    action = None
    for i in range(number_steps_ahead):
        prev_action = action
        action = agent.act(state)
        with torch.no_grad():
            _state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
            q_values = agent.online_model.forward(_state)
            q_values_softmax = torch.nn.functional.softmax(Q_VALUE_SENSITIVITY * q_values, dim=1)

            current_snapshot = env.ale.cloneState()
            for j in range(len(actions)):
                next_state, reward, done, info = env.step(j)
                actionTree.add_node(
                    '{}_{}'.format(str(i + 1), str(j)),
                    pos=(3 * (i + 1), (len(actions) - 1) / 2 - j),
                    image=env.ale.getScreenRGB()
                )
                actionTree.add_edge(
                    '0' if i == 0 else '{}_{}'.format(str(i), str(prev_action)),
                    '{}_{}'.format(str(i + 1), str(j)),
                    label='{}\n{}'.format(action_dict[actions[j]], round(q_values[0, j].item(), 2)),
                    width=max(SCALE * 4 * q_values_softmax[0, j].item(), 1)
                )
                env.ale.restoreState(current_snapshot)

            next_state, reward, done, info = env.step(action)
            next_state = agent.preProcess(next_state)  # Process image
            next_state = np.stack((next_state, state[0], state[1], state[2]))
            state = next_state
    pos = nx.get_node_attributes(actionTree, 'pos')
    fig = plt.figure(episode * MAX_STEP + step, figsize=(SCALE * 12, SCALE * 5))
    ax = fig.add_subplot(111)
    nx.draw(actionTree, pos=pos, width=list(nx.get_edge_attributes(actionTree, 'width').values()), node_size=0)
    nx.draw_networkx_edge_labels(actionTree, pos=pos, font_size=SCALE * 7,
                                 edge_labels=nx.get_edge_attributes(actionTree, 'label'))

    for i in range(number_steps_ahead):
        for j in range(len(actions)):
            coords = ax.transData.transform((3 * (i + 1), (len(actions) - 1) / 2 - j))
            fig.figimage(actionTree.nodes['{}_{}'.format(str(i + 1), str(j))]['image'], xo=coords[0] - 50,
                         yo=coords[1] - 80, zorder=1)

    env.ale.restoreState(snapshot)
    return fig


environment = gym.make(ENVIRONMENT)  # Get env
agent = pong.Agent(environment)  # Create Agent

if LOAD_MODEL_FROM_FILE:
    agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))

    with open(MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get('epsilon')

    startEpisode = LOAD_FILE_EPISODE + 1

else:
    startEpisode = 1

last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
total_step = 1  # Cumulkative sum of all steps in episodes

step = 0
episode = 0

startTime = time.time()  # Keep time
state = environment.reset()  # Reset env

state = agent.preProcess(state)  # Process image

# Stack state . Every state contains 4 time contionusly frames
# We stack frames like 4 channel image
state = np.stack((state, state, state, state))

total_max_q_val = 0  # Total max q vals
total_reward = 0  # Total reward for each episode
total_loss = 0  # Total loss for each episode


@app.callback([
    Output('game-screen', 'src'),
    Output('saliency-map', 'src'),
    Output('action-tree', 'src')
],
    Input('interval-component', 'n_intervals'))
def take_step(n):
    global agent
    global state
    global episode
    global step

    action_tree_fig = showActionTree(environment, agent, state, episode, step, 6)

    # Select and perform an action
    action = agent.act(state)  # Act
    next_state, reward, done, info = environment.step(action)  # Observe

    next_state = agent.preProcess(next_state)  # Process image

    # Stack state . Every state contains 4 time contionusly frames
    # We stack frames like 4 channel image
    next_state = np.stack((next_state, state[0], state[1], state[2]))

    # Move to the next state
    state = next_state  # Update state

    step += 1

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

        if SAVE_MODELS:
            outputPath = MODEL_PATH + "out" + '.txt'  # Save outStr to file
            with open(outputPath, 'a') as outfile:
                outfile.write(outStr + "\n")

        episode += 1
        step = 0

    action_tree_fig.savefig('action_tree.png')
    encoded_action_tree = base64.b64encode(open('action_tree.png', 'rb').read())

    environment.ale.saveScreenPNG('game_screen.png')
    encoded_game_screen = base64.b64encode(open('game_screen.png', 'rb').read())

    occlusion_img = agent.getOcclusionImage(state, method='Gaussian-Blur', mode=MODE, action=ACTION, threshold=THRESHOLD, size=SIZE, color=None, concurrent=CONCURRENT, metric=METRIC)
    cv2.imwrite('saliency_map.png', cv2.resize(255 * occlusion_img, (340, 240)))
    encoded_saliency_map = base64.b64encode(open('saliency_map.png', 'rb').read())

    return 'data:image/png;base64,{}'.format(encoded_game_screen.decode()), 'data:image/png;base64,{}'.format(encoded_saliency_map.decode()),'data:image/png;base64,{}'.format(
        encoded_action_tree.decode())


app.run_server(debug=True)
