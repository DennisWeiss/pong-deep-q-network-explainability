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
from pong import showActionTree
from pong import showActionTreeV3
import matplotlib

matplotlib.use('Agg')

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
THRESHOLD = 0.0
MODE = 'value'  # 'value' , 'action' or 'advantage', if not 'value', parameter action=ACTION needs to be valid
# 'value' refers to the value estimation in the network, advantage stands for the advantage estimation and 'action' stands for the final logits
# The MODE determines what values will be used to compute the occlusion maps
ACTION = -1  # set -1 if you want saliency map for the whole action advantage vector/whole output vector of the network
CHOSENACTION = False  # If this is true, ACTION will be updated each frame with the action that the agent chose last
CONCURRENT = False  # If true, all regions are occluded at the same time in the 4 frames. If false, seperate maps for each frame is generated.
METRIC = "Norm"  # What value to compute from logits
SIZE = 2.0

paused = False
action_tree_selection = 'best-strategies'

tuBerlinLogo = base64.b64encode(open('2000px-TU-Berlin-Logo.png', 'rb').read())

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

white_button_style = {'color': '#DED8D8', 'margin-right': '15x', 'margin-left': '15px'}

app = dash.Dash()

app.layout = html.Div(children=[
    # header
    html.Div(
        [
            html.Div(
                [
                    html.H4("Understanding Policies", className="app__header__title"),
                    html.P(
                        "A WebApp made by Galip Ümit Yolcu, Dennis Weiss and Egemen Okur to understand policies of reinforcement learning based agents.",
                        className="app__header__title--grey",
                    ),
                ],
                className="app__header__desc",
            ),
            html.Div(
                [
                    html.A(
                        html.Button("Github", className="link-button", style=white_button_style),
                        href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-wind-streaming",
                    ),
                    html.A(
                        html.Button("Paper", className="link-button", style=white_button_style),
                        href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-wind-streaming",
                    ),
                    html.A(
                        html.Img(
                            src='data:image/png;base64,{}'.format(tuBerlinLogo.decode()),
                            className="app__menu__img",
                        ),
                        href="https://plotly.com/dash/",
                    ),
                ],
                className="app__header__logo",
            ),
        ],
        className="app__header",
    ),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(children=[
                        html.Div(
                            [
                                html.H6(
                                    "Game Screen", className="graph__title"
                                )
                            ]
                        ),
                        html.Img(
                            id='game-screen',
                            style={'align': 'center',
                                   "padding": "0px 20px 10px 20px"}
                        )],
                        className="graph__container first",
                        style={'display': 'inline-block', 'margin': '8px'}
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "Saliency Map", className="graph__title"
                                    )
                                ]
                            ),
                            html.Img(
                                id='saliency-map',
                                style={'align': 'center',
                                       "padding": "0px 20px 10px 20px"}
                            )
                        ],
                        className="graph__container second",
                        style={'display': 'inline-block', 'margin': '8px'}
                    ),
                    html.Div(children=[
                        html.Img(
                            id='action-tree',
                            width=1500,
                            style={"padding": "10px 20px 10px 20px"}
                        )],
                        className="graph__container second",
                        style={'margin': '8px'}
                    ),
                    dcc.Interval(
                        id='interval-component',
                        interval=4000,
                        n_intervals=0
                    )
                ],
                style={'width': '85%', 'display': 'inline-block'}
            ),
            html.Div(
                children=[
                    html.Button(children='Test', id='play-and-pause', style={'margin': '0 auto'}, n_clicks=0),
                    dcc.RadioItems(
                        id='action-tree-selector',
                        options=[
                            {'label': 'Taken action', 'value': 'taken-action'},
                            {'label': 'Best strategies', 'value': 'best-strategies'},
                        ],
                        value='taken-action',
                        labelStyle={'display': 'inline-block', 'color': '#DED8D8',
                                    'margin-right': '15px', 'margin-top': '15px'}
                    ),
                ],
                style={'width': '15%', 'display': 'inline-block', 'padding-top': '40px'}
            )
        ],
        style={'display': 'flex'}
    )])

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


def pong_step(draw_explainability=True):
    global agent
    global state
    global episode
    global step
    global paused
    global action_tree_selection

    print(action_tree_selection)

    if paused:
        raise (Exception('Game is paused'))

    if draw_explainability:
        if action_tree_selection == 'taken-action':
            action_tree_fig = showActionTree(environment, agent, state, episode, step, 6)
        elif action_tree_selection == 'best-strategies':
            action_tree_fig = showActionTreeV3(environment, agent, state, episode, step, 4, 8)

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

    if draw_explainability:
        action_tree_fig.savefig('action_tree.png')
        encoded_action_tree = base64.b64encode(open('action_tree.png', 'rb').read())

        environment.ale.saveScreenPNG('game_screen.png')
        encoded_game_screen = base64.b64encode(open('game_screen.png', 'rb').read())

        occlusion_img = agent.getOcclusionImage(state, method='Gaussian-Blur', mode=MODE, action=ACTION,
                                                threshold=THRESHOLD, size=SIZE, color=None, concurrent=CONCURRENT,
                                                metric=METRIC)
        cv2.imwrite('saliency_map.png', cv2.resize(255 * occlusion_img, (340, 240)))
        encoded_saliency_map = base64.b64encode(open('saliency_map.png', 'rb').read())

        return 'data:image/png;base64,{}'.format(encoded_game_screen.decode()), 'data:image/png;base64,{}'.format(
            encoded_saliency_map.decode()), 'data:image/png;base64,{}'.format(encoded_action_tree.decode())


@app.callback(
    Output('play-and-pause', 'children'),
    [Input('play-and-pause', 'n_clicks')])
def clicks(n_clicks):
    global paused
    paused = n_clicks % 2 == 1
    return 'Resume' if n_clicks % 2 == 1 else 'Pause'


@app.callback([
    Output('game-screen', 'src'),
    Output('saliency-map', 'src'),
    Output('action-tree', 'src')
],
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def take_step(n):
    if n == 0:
        for i in range(25):
            pong_step(False)
    return pong_step(True)


@app.callback(
    Input('action-tree-selector', 'value')
)
def action_tree_select(value):
    global action_tree_selection
    print(value)
    action_tree_selection = value


if __name__ == '__main__':
    app.run_server(debug=True)
