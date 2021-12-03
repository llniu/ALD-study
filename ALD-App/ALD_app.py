import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import base64
import ast
import seaborn as sns
import plotly.graph_objs as go
from scipy import stats
import plotly.express as px

from config import FNAME_IMAGE1
from config import DATA_PL_MEDIAN, DATA_PLCORR, DATA_PLASMA_LONG, DATA_LIVER_LONG


# Prepare images
def png_encode(image_png):
    png_base64 = base64.b64encode(open(image_png, 'rb').read()).decode('ascii')
    return (png_base64)


image1_base64 = png_encode(FNAME_IMAGE1)

data_pl_median = pd.read_csv(DATA_PL_MEDIAN, index_col='Protein ID')
options_ploverlap = [{'value': i, 'label': i}
                     for i in list(data_pl_median['Genename_ProteinID'].unique())]

data_plcorr = pd.read_csv(DATA_PLCORR, index_col='Protein ID')

data_plasma_long = pd.read_csv(DATA_PLASMA_LONG, index_col='Genename_ProteinID')
data_liver_long = pd.read_csv(DATA_LIVER_LONG, index_col='Genename_ProteinID')
options_box_plasma_proteinID = [{'value': i, 'label': i}
                                for i in list(data_plasma_long.index)]
options_box_liver_proteinID =  [{'value': i, 'label': i}
                                for i in list(data_liver_long.index)]
options_histology_score = [{'value': i, 'label': i}
                           for i in ['kleiner', 'nas_inflam', 'nas_steatosis_ordinal']]
# Figures
template='simple_white'

figure_colorful = px.scatter(data_plcorr, x='plasma',y='liver', color='Genename_ProteinID', template=template, opacity=0.8)
figure_colorful.update_traces(marker={'size':7, 'line':dict(width=0.5, color='lightyellow')})
figure_colorful.update_layout(xaxis={'title': 'MS signal in plasma [Log10]'},
                              yaxis={'title': 'MS signal in liver [Log10]'},
                              title='The liver-plasma proteome space',
                              showlegend=False)
# Styles
style_headerh1 = {'backgroundColor': '#243E58', 'color': 'snow', 'textAlign': 'center',
                  'height': '120px', 'line-height': '120px', 'border': '2px solid white',
                  'font-style': 'normal', 'font-family': 'Copperplate', 'first-letter': {'color': 'red'}
                  }
style_graph_2panel = {'display': 'inline-block', 'width': '50%', 'textAlign': 'center'}
style_userinput = {'font-size': '120%', 'color': 'darkblue'}
tab_style = {'height': '80px', 'border': '1px solid black', 'backgroundColor': 'whitesmoke',
             'line-height': '40px', 'textAlign': 'center', 'font-size': '120%', 'color': '#243E58'}
selected_tab_style = {'height': '80px', 'border': 'none', 'backgroundColor': 'white',
                      'line-height': '40px', 'textAlign': 'center', 'font-size': '120%', 'color': '#243E58'}
################################################################################
# building Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1('Proteomics biomarker discovery in liver disease', style=style_headerh1),
    html.Div([
        dcc.Tabs(id='tabs', value='3', children=[
            dcc.Tab(id='tab1', label='Proteome-Histology Integration', value='1',
            children=[
                html.Div([
                    html.Div([
                        html.P('Select a histology score here:', style=style_userinput),
                        dcc.Dropdown(id='boxplot_histology_score_plasma',
                                     options=options_histology_score, value='kleiner'),
                        html.P('Select a protein here: (GeneName_ProteinID)',
                               style=style_userinput),
                        dcc.Dropdown(
                            id='boxplot_proteinID',
                            options=options_box_plasma_proteinID,
                            value='LGALS3BP__Q08380'),
                        dcc.Graph(id='box_plasma'),

                    ], style=style_graph_2panel),
                    html.Div([
                        html.P('Select a histology score here:', style=style_userinput),
                        dcc.Dropdown(id='boxplot_histology_score_liver',
                                     options=options_histology_score, value='kleiner'),
                        html.P('Select a protein here: (GeneName_ProteinID)',
                               style=style_userinput),
                        dcc.Dropdown(
                            id='boxplot_proteinID_liver',
                            options=options_box_liver_proteinID,
                            value='LGALS3BP__Q08380'),
                        dcc.Graph(id='box_liver'),

                    ], style=style_graph_2panel)
                ], )
            ],
            style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(id='tab2', label='Study Overview', value='2',
            children=[
                html.Div([
                    html.Br(),
                    html.Br(),
                    html.Img(src='data:image/png;base64,{}'.format(image1_base64),
                             style={
                        'height': '50%',
                        'width': '50%'})
                ], style={'width': '90%', 'textAlign': 'center', 'padding-left': '5%'})
            ], style=tab_style, selected_style=selected_tab_style),
            dcc.Tab(id='tab3', label='Liver-Plasma Proteme Integration', value='3',
            children=[
                html.Div([
                    html.Div([
                        html.Br(),
                        html.P('Search here: (GeneName_ProteinID)', style=style_userinput),
                        dcc.Dropdown(options=options_ploverlap, value='ALB__P02768'),
                        dcc.Graph(figure=figure_colorful)
                    ], style={'width': '60%', 'textAlign': 'center', 'padding-left': '20%'}),
                    html.Div([
                        html.Br(),
                        html.P('Search here: (GeneName_ProteinID)', style=style_userinput),
                        dcc.Dropdown(id='pl_median_scatter_input',
                                     options=options_ploverlap, value='CRP__P02741'),
                    ], style={'width': '60%', 'textAlign': 'center', 'padding-left': '20%'}),
                    html.Div([
                        html.Br(),
                        dcc.Graph(id='pl_median_scatter')
                    ], style=style_graph_2panel),
                    html.Div([
                        html.Br(),
                        dcc.Graph(id='pl_corr_scatter')
                    ], style=style_graph_2panel)
                ])
            ], style=tab_style, selected_style=selected_tab_style)
        ])
    ])
])


@app.callback(
    Output('pl_median_scatter', 'figure'),
    [Input('pl_median_scatter_input', 'value')]
)
def update_figure_plcorr_scatter(input_value):
    df = data_pl_median.copy()
    df = df.reset_index().set_index('Genename_ProteinID')
    df['color'] = 'other proteins'
    df.loc[input_value, 'color'] = 'selected protein'
    figure = px.scatter(df, x='plasma', y='liver', template=template,
                        color='color',
                        color_discrete_sequence=['deepskyblue', 'darkred'],
                        opacity=0.8)
    figure.update_traces(marker={'size':12, 'line':dict(color='black', width=1.5)})
    figure.update_layout(title='Liver-Plasma proteome correlation', xaxis={'title': 'Protein intensity in plasma [Log10]'},
                         yaxis={'title': 'Protein intensity in liver [Log10]'},
                         width=500,
                         showlegend=False)

    return figure


@app.callback(
    Output('pl_corr_scatter', 'figure'),
    [Input('pl_median_scatter_input', 'value')]
)
def update_figure_plcorr_scatter(input_value):
    # df = data_plcorr.copy()
    df = data_plcorr[data_plcorr['Genename_ProteinID'] == input_value]
    fig = px.scatter(df, x='plasma', y='liver', template=template,
                    color_discrete_sequence=['deepskyblue'],
                    opacity=0.8)
    fig.update_traces(marker={'size':12, 'line':dict(color='black', width=1.5)})
    fig.update_layout(xaxis={'title': 'MS signal in plasma [Log10]'},
                      yaxis={'title': 'MS signal in liver [Log10]'},
                      width=500,
                      title=f'Pair-wise correlation: {input_value}')
    return fig


@app.callback(
    dash.dependencies.Output('box_plasma', 'figure'),
    [dash.dependencies.Input('boxplot_histology_score_plasma', 'value'),
     dash.dependencies.Input('boxplot_proteinID', 'value')]
)
def figure_box_plot_plasma(histologyscore, proteinID):
    df = data_plasma_long.loc[proteinID]
    df['mcolor']='deepskyblue'
    figure = px.box(df, x=histologyscore, y="Intensity", notched=True, points='all', template=template,
                    color='mcolor', color_discrete_sequence=['steelblue'])
    figure.update_layout(yaxis={'title': 'Protein intensity [Log2]'},
                         title='Protein levels in plasma',
                         showlegend=False)

    return figure

@app.callback(
    dash.dependencies.Output('box_liver', 'figure'),
    [dash.dependencies.Input('boxplot_histology_score_liver', 'value'),
     dash.dependencies.Input('boxplot_proteinID_liver', 'value')]
)
def figure_box_plot_liver(histologyscore, proteinID):
    df = data_liver_long.loc[proteinID]
    df['mcolor']='deepskyblue'
    figure = px.box(df, x=histologyscore, y="Intensity", notched=False, points='all', template=template,
                    color='mcolor', color_discrete_sequence=['steelblue'])
    figure.update_layout(yaxis={'title': 'Protein intensity [Log2]'},
                         title='Protein levels in the liver',
                         showlegend=False)

    return figure

if __name__ == '__main__':
    DEBUG = False
    app.run_server(debug=DEBUG)
