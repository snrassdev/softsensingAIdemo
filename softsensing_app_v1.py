# Adding Tab 3 and Tab 4: Feature Engineering + NH3 & CO2 Prediction with download

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server
app.title = "Blusense Soft Sensing AI"

app.layout = html.Div([
    dcc.Store(id="stored-data", storage_type="session"),
    dcc.Store(id="stored-params", storage_type="session"),
    dcc.Store(id="predicted-data", storage_type="session"),
    
    dcc.Tabs(id="tabs", value='tab-upload', children=[
        dcc.Tab(label='1. Upload Trip Data', value='tab-upload'),
        dcc.Tab(label='2. Data Overview', value='tab-overview'),
        dcc.Tab(label='3. Feature Engineering', value='tab-feature'),
        dcc.Tab(label='4. NH₃ & CO₂ Prediction', value='tab-predict'),
    ]),

    html.Div(id='tab-content'),

    # ✅ Always present: Graphs to be updated via callbacks
    html.Div([
        html.Div(id="prediction-container", children=[
            html.H3("Soft Sensing Prediction: NH₃ & CO₂"),
            html.H5("Predicted NH₃ Levels (Smoothed)"),
            dcc.Loading(dcc.Graph(id='nh3-graph')),
            html.H5("Predicted CO₂ Levels (Smoothed)"),
            dcc.Loading(dcc.Graph(id='co2-graph')),
            html.Br(),
            html.A("⬇️ Download Predicted Data", id="download-link", download="predicted_data.xlsx",
                   href="", target="_blank", className="btn btn-success")
        ])
    ])
])


@app.callback(
    Output('tab-content', 'children'),
    Output('prediction-container', 'style'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-upload':
        return upload_tab_layout(), {"display": "none"}
    elif tab == 'tab-overview':
        return overview_tab_layout(), {"display": "none"}
    elif tab == 'tab-feature':
        return feature_tab_layout(), {"display": "none"}
    elif tab == 'tab-predict':
        return html.Div(), {"display": "block"}
    return html.Div("Tab not implemented."), {"display": "none"}


def upload_tab_layout():
    return dbc.Container([
        html.H2("Upload Trip Data (.xlsx)", className="mt-3"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['📤 Drag and Drop or ', html.A('Select Excel File')]),
            style={'width': '100%', 'height': '80px', 'lineHeight': '80px', 'borderWidth': '1px',
                   'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
            multiple=False
        ),
        html.Div(id='upload-status'),
        html.Hr(),
        html.H4("Enter Conditional Parameters"),
        dbc.Row([
            dbc.Col(dbc.Input(id='protein-feed', type='number', placeholder="Protein Feed %"), md=3),
            dbc.Col(dbc.Input(id='time-elapsed', type='number', placeholder="Time Since Feeding (hrs)"), md=3),
            dbc.Col(dbc.Input(id='stock-density', type='number', placeholder="Stock Density (kg)"), md=3),
            dbc.Col(dbc.Input(id='tank-volume', type='number', placeholder="Tank Volume (L)"), md=3),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Input(id='tank-filled', type='number', placeholder="Tank Filled %"), md=3),
            dbc.Col(dbc.Input(id='organic-decay', type='number', placeholder="Organic Matter Decay Factor"), md=3),
            dbc.Col(dbc.Input(id='nitrification-rate', type='number', placeholder="Nitrification Rate"), md=3),
            dbc.Col(dbc.Input(id='nh3-reabsorption', type='number', placeholder="NH₃ Reabsorption in Water"), md=3),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Input(id='water-air', type='number', placeholder="Water-Air Exchange"), md=3),
            dbc.Col(dbc.Button("Save Parameters", id="save-params", color="primary"), md=2)
        ]),
        html.Div(id='save-status', className="mt-2")
    ], fluid=True)

def overview_tab_layout():
    return dbc.Container([
        html.H3("Data Overview"),
        html.Div(id="data-preview"),
        html.Hr(),
        html.H5("Feature Plots Over Time"),
        dcc.Loading(dcc.Graph(id='feature-plots')),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Page Number"),
                dcc.Input(id='page-number', type='number', min=1, step=1, value=1),
            ], width=3),
        ])
    ], fluid=True)

def feature_tab_layout():
    return dbc.Container([
        html.H3("Feature Engineering Visualization"),
        dcc.Loading(dcc.Graph(id='feature-engineering-graph')),
    ], fluid=True)

def prediction_tab_layout():
    return dbc.Container([
        html.H3("Soft Sensing Prediction: "),
        html.H5("Predicted NH₃ Levels (Smoothed)"),
        dcc.Loading(dcc.Graph(id='nh3-graph')),
        html.H5("Predicted CO₂ Levels (Smoothed)"),
        dcc.Loading(dcc.Graph(id='co2-graph')),
        html.Br(),
        html.A("⬇️ Download Predicted Data", id="download-link", download="predicted_data.xlsx",
               href="", target="_blank", className="btn btn-success")
    ], fluid=True)

@app.callback(
    Output('upload-status', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return "", dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        return f"✅ File '{filename}' uploaded successfully.", df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return f"❌ Error: {str(e)}", dash.no_update

@app.callback(
    Output('save-status', 'children'),
    Output('stored-params', 'data'),
    Input('save-params', 'n_clicks'),
    State('protein-feed', 'value'), State('time-elapsed', 'value'),
    State('stock-density', 'value'), State('tank-volume', 'value'),
    State('tank-filled', 'value'), State('organic-decay', 'value'),
    State('nitrification-rate', 'value'), State('nh3-reabsorption', 'value'),
    State('water-air', 'value')
)
def save_parameters(n_clicks, pf, te, sd, tv, tf, od, nr, nh3r, wa):
    if not n_clicks:
        raise PreventUpdate
    params = {
        "Protein Feed %": pf, "Time Elapsed Since Feeding": te, "Stock Density": sd,
        "Tank Volume": tv, "Tank Filled %": tf, "Organic Matter Decay Factor": od,
        "Nitrification Rate": nr, "NH3 Ammonia Reabsorption Water": nh3r,
        "Water-Air Exchange": wa
    }
    return "✅ Parameters saved.", params

@app.callback(
    Output('data-preview', 'children'),
    Output('feature-plots', 'figure'),
    Input('page-number', 'value'),
    State('stored-data', 'data')
)
def update_overview(page_number, data_json):
    if data_json is None:
        raise PreventUpdate
    df = pd.read_json(data_json, orient='split')
    columns_to_display = [col for col in df.columns if col.lower() != 'ammonia']
    page_size = 50
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    preview_table = dash_table.DataTable(
        data=df[columns_to_display].iloc[start_idx:end_idx].to_dict('records'),
        columns=[{"name": i, "id": i} for i in columns_to_display],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    )
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("pH", "Temperature", "TDS", "DO"))
    if 'ph' in df.columns:
        fig.add_trace(go.Scatter(y=df['ph'], name='pH'), row=1, col=1)
    if 'temp' in df.columns:
        fig.add_trace(go.Scatter(y=df['temp'], name='Temp'), row=1, col=2)
    if 'tds' in df.columns:
        fig.add_trace(go.Scatter(y=df['tds'], name='TDS'), row=2, col=1)
    if 'resdo' in df.columns:
        fig.add_trace(go.Scatter(y=df['resdo'], name='DO'), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return preview_table, fig

@app.callback(
    Output('feature-engineering-graph', 'figure'),
    Input('stored-data', 'data')
)
def show_feature_engineering(data_json):
    if data_json is None:
        raise PreventUpdate

    df = pd.read_json(data_json, orient='split')

    # Create new features
    df['pH_Temp'] = df['ph'] * df['temp']
    df['TDS_DO'] = df['tds'] * df['resdo']
    df['pH_TDS'] = df['ph'] * df['tds']
    df['Stability_Index'] = (df['tds'] / df['resdo'].replace(0, np.nan)) * df['ph']

    # Remove NaNs or infs (to avoid scaling issues)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Scale engineered features using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = ['pH_Temp', 'TDS_DO', 'pH_TDS', 'Stability_Index']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    # Plot
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=features)
    fig.add_trace(go.Scatter(y=df_scaled['pH_Temp'], name='pH×Temp'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df_scaled['TDS_DO'], name='TDS×DO'), row=1, col=2)
    fig.add_trace(go.Scatter(y=df_scaled['pH_TDS'], name='pH×TDS'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df_scaled['Stability_Index'], name='Stability'), row=2, col=2)

    fig.update_layout(height=600, title="Feature Engineering Visualization (Scaled 0–1)", showlegend=False)
    return fig

@app.callback(
    Output('nh3-graph', 'figure'),
    Output('co2-graph', 'figure'),
    Output('predicted-data', 'data'),
    Input('tabs', 'value'),
    State('stored-data', 'data')
)
def predict_nh3_co2(tab, data_json):
    if tab != 'tab-predict' or data_json is None:
        raise PreventUpdate

    df = pd.read_json(data_json, orient='split')
    df['pH_Temp'] = df['ph'] * df['temp']
    df['TDS_DO'] = df['tds'] * df['resdo']
    df['pH_TDS'] = df['ph'] * df['tds']
    df['Stability_Index'] = (df['tds'] / df['resdo']) * df['ph']

    features = ['ph', 'temp', 'resdo', 'tds', 'pH_Temp', 'TDS_DO', 'pH_TDS', 'Stability_Index']
    X = df[features]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled[..., np.newaxis]

    model_nh3 = tf.keras.models.load_model('vt1_cnn_modelnh3.h5')
    model_co2 = tf.keras.models.load_model('vt1_cnn_model_co2.h5')

    nh3_pred = model_nh3.predict(X_scaled).flatten()
    co2_pred = model_co2.predict(X_scaled).flatten()

    df = df.iloc[-len(nh3_pred):].copy()
    df['Predicted_NH3'] = nh3_pred
    df['Predicted_CO2'] = co2_pred

    ma_nh3 = pd.Series(nh3_pred).rolling(100, min_periods=1).mean()
    ma_co2 = pd.Series(co2_pred).rolling(100, min_periods=1).mean()

    fig_nh3 = go.Figure()
    fig_nh3.add_trace(go.Scatter(y=ma_nh3, name="NH₃", line=dict(color='orange')))
    fig_nh3.update_layout(title="Predicted NH₃ ", xaxis_title="Elapsed Time", yaxis_title="NH₃ Level")

    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(y=ma_co2, name="CO₂", line=dict(color='red')))
    fig_co2.update_layout(title="Predicted CO₂", xaxis_title="Elapsed Time", yaxis_title="CO₂ Level")

    return fig_nh3, fig_co2, df.to_json(date_format='iso', orient='split')


@app.callback(
    Output("download-link", "href"),
    Input("predicted-data", "data")
)
def generate_download_link(data_json):
    if not data_json:
        raise PreventUpdate
    df = pd.read_json(data_json, orient='split')
    out_buffer = io.BytesIO()
    df.to_excel(out_buffer, index=False, engine='openpyxl')
    encoded = base64.b64encode(out_buffer.getvalue()).decode()
    return "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + encoded

if __name__ == '__main__':
    app.run(debug=True)
