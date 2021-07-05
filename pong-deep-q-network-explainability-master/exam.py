import os
import pathlib
import numpy as np
import datetime as dt
import dash
import base64
import dash_core_components as dcc
import dash_html_components as html



from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 5000)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Understanding Policies"

img1 = base64.b64encode(open('action_tree.png', 'rb').read())
img2 = base64.b64encode(open('saliency_map.png', 'rb').read())
img3 = base64.b64encode(open('game_screen.png', 'rb').read())
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
                            [html.H6("Explanation about the project",className="graph__title"),
                             html.H6("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",className="text__container")
                             ]
                        ),
                    ],
                    className="one-third column first__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [html.H6("Explanation of Neural Network",className="graph__title"),
                             html.H6("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",className="text__container")


                             ]
                        ),
                    ],
                    className="one-third column second__container",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                             html.H6("Model of Neural Network", className="graph__title"),
                             html.H6("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",className="text__container")

                             ]
                        ),
                    ],
                    className="one-third column third__container",
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
                        dcc.RadioItems(
                            options=[
                                {'label': 'Checkbox1', 'value': 'Check1'},
                                {'label': 'Checkbox2', 'value': 'Check2'},
                            ],
                            value='MTL',
                            labelStyle={'display': 'inline-block','color':'#DED8D8', 'margin-left':'200px','margin-right':'15px','margin-top':'15px'}
                        ),
                        html.Img(



                            src='data:image/png;base64,{}'.format(img1.decode()),
                            style={'height': '82%', 'width': '95%', 'display': 'inline-block', 'text-align': 'center', "padding": "25px 25px 0px 25px" }
                            # className="app__menu__img",
                        ),
                    ],
                    className="two-thirds column wind__speed__container",
                ),
                html.Div(
                    [
                        # histogram
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Game FPS",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        dcc.Slider(
                                            id="bin-slider",
                                            min=1,
                                            max=60,
                                            step=1,
                                            value=20,
                                            updatemode="drag",
                                            marks={
                                                20: {"label": "20"},
                                                40: {"label": "40"},
                                                60: {"label": "60"},
                                            },
                                        )
                                    ],
                                    className="slider",
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="bin-auto",
                                            options=[
                                                {"label": "Auto", "value": "Auto"}
                                            ],
                                            value=["Auto"],
                                            inputClassName="auto__checkbox",
                                            labelClassName="auto__label",
                                        ),
                                        html.P(
                                            "Here comes another label",
                                            id="bin-size",
                                            className="auto__p",
                                        ),
                                    ],
                                    className="auto__container",
                                ),
                                html.A(
                                    html.Img(
                                        src='data:image/png;base64,{}'.format(img3.decode()),
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',"padding": "0px 20px 10px 20px" }
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # wind direction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Saliency Map", className="graph__title"
                                        )
                                    ]
                                ),
                                html.A(
                                    html.Img(
                                        src='data:image/png;base64,{}'.format(img2.decode()),
                                        style={'width': '90%', 'display': 'inline-block', 'align': 'center',"padding": "0px 20px 10px 20px" }
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
        # Footer
        html.Div(
            [

                html.Div(
                    [
                        html.H4("Conclusion", className="app__header__title",  style={"margin-left": "15px" }),
                        html.H6(
                            "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",
                            className="text__container",)

                    ],
                    className="one-whole column footer__container",
                ),
            ],
            className="app__header",
        ),
    ],
    className="app__container",
)

if __name__ == "__main__":
    app.run_server(debug=True)

