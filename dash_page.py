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

# OCCLUSION PARAMETERS
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

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Visualising Policies of Agents in Reinforcement Learning"

img1 = base64.b64encode(open('action_tree.png', 'rb').read())
img2 = base64.b64encode(open('model.png', 'rb').read())
img3 = base64.b64encode(open('saliency.png', 'rb').read())
img4 = base64.b64encode(open('2000px-TU-Berlin-Logo.png', 'rb').read())

server = app.server

app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}

white_button_style = {'color': '#DED8D8', 'margin-right':'15x', 'margin-left':'15px'}
app.layout = html.Div(
    [




        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Understanding Policies of Agents in Reinforcement Learning", className="app__header__title"),
                        html.P(
                            "A WebApp made by Galip Ümit Yolcu, Dennis Weiss and Egemen Okur to understand policies of reinforcement learning based agents.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        # html.A(
                        #     html.Button("Github", className="link-button", style=white_button_style),
                        #     href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-wind-streaming",
                        # ),
                        # html.A(
                        #     html.Button("Paper", className="link-button", style=white_button_style),
                        #     href="https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-wind-streaming",
                        # ),
                        html.A(
                            html.Img(
                                src='data:image/png;base64,{}'.format(img4.decode()),
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
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("About",className="graph__title"),
                             html.H6("Reinforcement learning agents have achieved state-of-the-art results in ATARI games and are effective at maximizing rewards. Despite their impressive performance, they have been a black box. Understanding the agent's actions are important to interpret their models before using them to solve real-world problems. In this work, we investigate a deep RL agent that uses raw visual input to make their decisions in the Pong game. We have implemented a visualisation using the game theoretical concept of game tree. We have also implemented saliency maps with various methods.",className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column first__container",
                ),

                html.Div(
                    [
                        html.Div(
                            [
                             html.H6("Dueling Neural Network [1]", className="graph__title"),
                             html.H6("Instead of having a single stream of linear layers after the convolutional layers, this architecture provides seperate estimations of the state value function V(s) and the advantages for each action, which are combined in a special final layer to give the final Q(s), estimate of the Q values. The agent takes the last 4 ATARI frames in grayscale as input.",
                                     className="text__container"),
                            html.Img(
                                src='data:image/png;base64,{}'.format(img2.decode()),
                                style={'height': '40%', 'width': '90%', 'display': 'inline-block',
                                       'text-align': 'center', "padding": "0px 0px 0px 25px"}
                            )

                            ]
                        ),
                    ],
                    className="one-third column third__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [html.H6("Game Tree Visualisation", className="graph__title"),
                             html.H6(
                                 "At each state, the agent has six possible actions: no action(x), fire(o), joystick left(<-), joystick right(->), and fire+joystick left/right (<-o/o->). As each action results in a new state, possible evolutions of the game starting from a single initial state produces a tree. This tree is called the game tree. We have visualised the path that the agent takes in the game tree. At each level, we show for all actions, the Q-Value for that action as estimated by the agent. The thickness of the edges correspond to the agent's probability of taking that action, its trust in this action. We can see in this examplary state that the agent has learned enough to pass some sanity checks.",
                                 className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column second__container",
                ),
            ],
            className="app__content",
        ),
        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Game Tree with Q-Values", className="graph__title")]
                        ),
                        html.Img(
                            src='data:image/png;base64,{}'.format(img1.decode()),
                            style={'height': '82%', 'width': '95%', 'display': 'inline-block', 'text-align': 'center', "padding": "25px 25px 0px 25px" }
                        ),
                        dcc.Interval(
                            id='interval-component',
                            interval=4000,
                            n_intervals=0
                        )
                    ],
                    className="three-thirds column wind__speed__container",
                )

            ],
            className="app__content",
        ),

        html.Div(
            [
                # wind speed
                html.Div(
                    [
                        html.Div(
                            [html.H6("Saliency Maps", className="graph__title")]
                        ),
                        # dcc.RadioItems(
                        #     id='action-tree-selector',
                        #     options=[
                        #         {'label': 'Taken action', 'value': 'taken-action'},
                        #         {'label': 'Best strategies', 'value': 'best-strategies'},
                        #     ],
                        #     value='taken-action',
                        #     labelStyle={'display': 'inline-block', 'color': '#DED8D8',
                        #                 'margin-left': '45px', 'margin-top': '15px'}),
                        # html.Img(
                        #     id='action-tree',
                        #     style={'height': '82%', 'width': '95%', 'display': 'inline-block', 'text-align': 'center', "padding": "25px 25px 0px 25px" }
                        #     # className="app__menu__img",
                        # ),
                        html.Img(
                            src='data:image/png;base64,{}'.format(img3.decode()),
                            style={'height': '82%', 'width': '95%', 'display': 'inline-block', 'text-align': 'center',
                                   "padding": "25px 25px 0px 25px"}
                        )
                    ],
                    className="three-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Saliency Maps", className="graph__title"
                                        ),
                                        html.H6("Originally proposed for image classifying CNNs, the purpose of saliency maps is to represent the importance of pixels for the activation of a neuron in the network(mostly for output neurons) at every location in the input image by a scalar quantity. This  allows to show what parts of an image or video frame are most important to the network’s decisions.",
                                                className="text__container"
                                        ),

                                        html.H6(
                                            " We are showing results for the value estimate V(s), because all choices result in qualitatively equivalent saliency maps. Our agent uses the last 4 frames of the ATARI game as input, so we get 4 saliency maps for each method, except Gaussian Blur Perturbation method. This method has high computational complexity so we perturbed all 4 images at the same time for each pixel.",
                                            className="text__container"

                                        )

                                    ]
                                ),
                                html.A(
                                    html.Img(
                                        id='spacer1',
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',"padding": "0px 20px 10px 20px" }
                                    ),

                                ),
                            ],
                            className="graph__container first",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Conclusion", className="graph__title"
                                        ),
                                        html.H6(
                                            "We have seen that the agent can pass sanity checks (such as assigning similar values to actions '->/<-' and 'o->/<-o'. It also assigns similar values to all actions when the ball is moving away because the decision for what action to take is less important in those situations.",
                                            className="text__container"
                                        ),
                                        html.H6(
                                            "In all saliency maps, we see that the agent focuses mostly on the recent frames and least on the earliest frame in time. The gradient saliency maps are intruiguing, it is not clear what they mean. Some investigation of why this behaviour emerges and how general it is can be subject to future work. One interpretation of a highly active gradient map is that the agent is very vulnerable against adversarial attacks, as small changes in many of the pixels result in a big change in the output.",
                                            className="text__container"
                                        ),
                                        html.H6(
                                            "The Gaussian Blur method seems to be the most interpretable method but it is not clear how informative it is, because at many points the blurring does not change the input images much to make a practical difference.",
                                            className="text__container"
                                        )

                                    ]
                                ),
                                html.A(
                                    html.Img(
                                        id='spacer2',
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',
                                               "padding": "0px 20px 10px 20px"}
                                    ),

                                ),
                            ],
                            className="graph__container second",
                        ),

                    ],
                    className="one-third column histogram__direction",
                ),

            ],
            className="app__content",
        ),



        html.Div(
            [

                html.Div(
                    [
                        html.H4("References", className="app__header__title",  style={"margin-left": "15px" }),
                        html.H6(
                            '[1] Wang, Ziyu, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. "Dueling network architectures for deep reinforcement learning." In International conference on machine learning, pp. 1995-2003. PMLR, 2016.',
                            className="text__container"
                        ),
                        html.H6(
                            '[2] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013).',
                            className="text__container"
                        ),
                        html.H6(
                            '[3] Springenberg, Jost Tobias, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. "Striving for simplicity: The all convolutional net." arXiv preprint arXiv:1412.6806 (2014).',
                            className="text__container"
                        ),
                        html.H6(
                            '[4] Greydanus, Samuel, Anurag Koul, Jonathan Dodge, and Alan Fern. "Visualizing and understanding atari agents." In International Conference on Machine Learning, pp. 1792-1801. PMLR, 2018.',
                            className="text__container"
                        )

                    ],
                    className="one-whole column footer__container",
                ),
            ],
            className="app__header",
        ),
    ],
    className="app__container",
)



#
#
# environment = gym.make(ENVIRONMENT)  # Get env
# agent = pong.Agent(environment)  # Create Agent
#
# if LOAD_MODEL_FROM_FILE:
#     agent.online_model.load_state_dict(torch.load(MODEL_PATH + str(LOAD_FILE_EPISODE) + ".pkl", map_location="cpu"))
#
#     with open(MODEL_PATH + str(LOAD_FILE_EPISODE) + '.json') as outfile:
#         param = json.load(outfile)
#         agent.epsilon = param.get('epsilon')
#
#     startEpisode = LOAD_FILE_EPISODE + 1
#
# else:
#     startEpisode = 1
#
# last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
# total_step = 1  # Cumulkative sum of all steps in episodes
#
# step = 0
# episode = 0
#
# startTime = time.time()  # Keep time
# state = environment.reset()  # Reset env
#
# state = agent.preProcess(state)  # Process image
#
# # Stack state . Every state contains 4 time contionusly frames
# # We stack frames like 4 channel image
# state = np.stack((state, state, state, state))
#
# total_max_q_val = 0  # Total max q vals
# total_reward = 0  # Total reward for each episode
# total_loss = 0  # Total loss for each episode


# def pong_step(draw_explainability=True):
#     global agent
#     global state
#     global episode
#     global step
#     global paused
#     global action_tree_selection
#
#     if paused:
#         raise (Exception('Game is paused'))
#
#     if draw_explainability:
#         if action_tree_selection == 'taken-action':
#             action_tree_fig = showActionTree(environment, agent, state, episode, step, 6)
#         elif action_tree_selection == 'best-strategies':
#             action_tree_fig = showActionTreeV3(environment, agent, state, episode, step, 4, 8)
#
#     # Select and perform an action
#     action = agent.act(state)  # Act
#     next_state, reward, done, info = environment.step(action)  # Observe
#
#     next_state = agent.preProcess(next_state)  # Process image
#
#     # Stack state . Every state contains 4 time contionusly frames
#     # We stack frames like 4 channel image
#     next_state = np.stack((next_state, state[0], state[1], state[2]))
#
#     # Move to the next state
#     state = next_state  # Update state
#
#     step += 1
#
#     if done:  # Episode completed
#         currentTime = time.time()  # Keep current time
#         time_passed = currentTime - startTime  # Find episode duration
#         current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
#         epsilonDict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file
#
#         if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  # Save model as file
#             weightsPath = MODEL_PATH + str(episode) + '.pkl'
#             epsilonPath = MODEL_PATH + str(episode) + '.json'
#
#             torch.save(agent.online_model.state_dict(), weightsPath)
#             with open(epsilonPath, 'w') as outfile:
#                 json.dump(epsilonDict, outfile)
#
#         if TRAIN_MODEL:
#             agent.target_model.load_state_dict(agent.online_model.state_dict())  # Update target model
#
#         last_100_ep_reward.append(total_reward)
#         avg_max_q_val = total_max_q_val / step
#
#         outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
#             episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val,
#             agent.epsilon, time_passed, step, total_step
#         )
#
#         if SAVE_MODELS:
#             outputPath = MODEL_PATH + "out" + '.txt'  # Save outStr to file
#             with open(outputPath, 'a') as outfile:
#                 outfile.write(outStr + "\n")
#
#         episode += 1
#         step = 0
#
#     if draw_explainability:
#         action_tree_fig.savefig('action_tree.png')
#         encoded_action_tree = base64.b64encode(open('action_tree.png', 'rb').read())
#
#         environment.ale.saveScreenPNG('game_screen.png')
#         encoded_game_screen = base64.b64encode(open('game_screen.png', 'rb').read())
#
#         occlusion_img = agent.getOcclusionImage(state, method='Gaussian-Blur', mode=MODE, action=ACTION,
#                                                 threshold=THRESHOLD, size=SIZE, color=None, concurrent=CONCURRENT,
#                                                 metric=METRIC)
#         cv2.imwrite('saliency_map.png', cv2.resize(255 * occlusion_img, (340, 240)))
#         encoded_saliency_map = base64.b64encode(open('saliency_map.png', 'rb').read())
#
#         return 'data:image/png;base64,{}'.format(encoded_game_screen.decode()), 'data:image/png;base64,{}'.format(
#             encoded_saliency_map.decode()), 'data:image/png;base64,{}'.format(encoded_action_tree.decode())
#
#
# @app.callback(
#     Output('play-and-pause', 'children'),
#     [Input('play-and-pause', 'n_clicks')])
# def clicks(n_clicks):
#     global paused
#     paused = n_clicks % 2 == 1
#     return 'Resume' if n_clicks % 2 == 1 else 'Pause'
#
#
# @app.callback([
#     Output('game-screen', 'src'),
#     Output('saliency-map', 'src'),
#     Output('action-tree', 'src')
# ],
#     [Input('interval-component', 'n_intervals')]
# )
# def take_step(n):
#      if n == 0:
#          for i in range(25):
#              pong_step(False)
#      return pong_step(True)
#
#
# @app.callback(
#     Output('action-tree-selector', 'children'),
#     [Input('action-tree-selector', 'value')]
# )
# def action_tree_select(value):
#     global action_tree_selection
#     action_tree_selection = value
#     return []
#

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_ui=False)
