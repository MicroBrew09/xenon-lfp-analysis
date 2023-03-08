from tkinter import NW
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.graph_objects as go
import dash_daq as daq
from dash import dash_table
import json
from scipy import signal
from typing import Optional
import numpy as np
import h5py
import pandas as pd
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import os
import scipy.io
from datetime import datetime


def parameter(h5,typ):
    if typ == 'bw4':

        parameters = {}
        parameters['Ver'] = 'BW4'
        
        parameters['nRecFrames'] = h5['/3BRecInfo/3BRecVars/NRecFrames'][0]
        parameters['samplingRate'] = h5['/3BRecInfo/3BRecVars/SamplingRate'][0]
        parameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
        parameters['signalInversion'] = h5['/3BRecInfo/3BRecVars/SignalInversion'][0]  # depending on the acq version it can be 1 or -1
        parameters['maxUVolt'] = h5['/3BRecInfo/3BRecVars/MaxVolt'][0]  # in uVolt
        parameters['minUVolt'] = h5['/3BRecInfo/3BRecVars/MinVolt'][0]  # in uVolt
        parameters['bitDepth'] = h5['/3BRecInfo/3BRecVars/BitDepth'][0]  # number of used bit of the 2 byte coding
        parameters['qLevel'] = 2 ^ parameters['bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
        parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
        try:
            parameters['recElectrodeList'] = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]  # list of the recorded channels
            parameters['Typ'] = 'RAW'
        except:
            parameters['recElectrodeList']= h5['/3BRecInfo/3BMeaStreams/WaveletCoefficients/Chs'][:]
            parameters['Typ'] = 'WAV'
        parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])
    
    else:
        
        if "Raw" in h5['Well_A1'].keys():
            json_s = json.loads(h5['ExperimentSettings'][0].decode('utf8'))
            parameters = {}
            parameters['Ver'] = 'BW5'
            parameters['Typ'] = 'RAW'
            parameters['nRecFrames'] = h5['Well_A1/Raw'].shape[0]//4096
            parameters['samplingRate'] = json_s['TimeConverter']['FrameRate']
            parameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
            parameters['signalInversion'] = int(1)  # depending on the acq version it can be 1 or -1
            parameters['maxUVolt'] = int(4125)  # in uVolt
            parameters['minUVolt'] = int(-4125)  # in uVolt
            parameters['bitDepth'] = int(12)  # number of used bit of the 2 byte coding
            parameters['qLevel'] = 2 ^ parameters['bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
            parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
            parameters['recElectrodeList'] = getChMap()[:]  # list of the recorded channels
            parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])
        else: 
            json_s = json.loads(h5['ExperimentSettings'][0].decode('utf8'))
            parameters = {}
            parameters['Ver'] = 'BW5'
            parameters['Typ'] = 'WAV'
            parameters['nRecFrames'] = int(h5['Well_A1/WaveletBasedEncodedRaw'].shape[0]//4096
                                            //json_s['DataSettings']['WaveletBasedRawCoefficients']['FramesChunkSize']
                                            *json_s['TimeConverter']['FrameRate'])
            parameters['samplingRate'] = json_s['TimeConverter']['FrameRate']
            parameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
            parameters['signalInversion'] = int(1)  # depending on the acq version it can be 1 or -1
            parameters['maxUVolt'] = int(4125)  # in uVolt
            parameters['minUVolt'] = int(-4125)  # in uVolt
            parameters['bitDepth'] = int(12)  # number of used bit of the 2 byte coding
            parameters['qLevel'] = 2 ^ parameters['bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
            parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
            parameters['recElectrodeList'] = getChMap()[:]  # list of the recorded channels
            parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])

    return parameters


def check_filename(path):
    #h5 = h5py.File(path, 'r')
    #parameters = parameter(h5)
    try:
        h5 = h5py.File(path, 'r')

        typ = ''
        if 'ExperimentSettings' in h5.keys(): 
            typ = 'bw5'
            parameters = parameter(h5,'bw5')
            #print(parameters)
        elif '/3BRecInfo/3BRecVars/NRecFrames' in h5.keys():
            typ = 'bw4'
            parameters = parameter(h5,'bw4')
            #print(parameters)
        else:
            typ = 'File Not Recognized'
            #print(typ)
        return True, parameters
    except:
        parameters = {}
        parameters['Ver'] = "Not a BRW Recording"
        parameters['Typ'] = "Unrecognized File"
        return False, parameters



def get_grid(Xs,Ys,xs,ys):
    fig2 = go.Figure()
    x_label = np.linspace(1, 64, 64)
    y_label = np.linspace(1, 64, 64)
    xx, yy = np.meshgrid(x_label, y_label, sparse=False, indexing='xy')
    fig2.add_trace(go.Scatter(x=xs, y=ys, opacity = 0.1, marker={'color': 'grey', 'showscale': False}, mode='markers'))
    fig2.add_trace(go.Scatter(x=Xs, y=Ys, marker={'color': 'green', 'showscale': False}, mode='markers'))
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65],autorange="reversed")
    fig2.update_layout(template="plotly_white", width=600, height=600,showlegend = False, legend=dict(orientation="h"),margin=dict(l=25,r=10,b=10,t=10,pad=4))

    return fig2

def get_chMap(chs):
    Ys = []
    Xs = []
    idx = []
    
    for n, item in enumerate(chs):
        Ys.append(item['Col'])
        Xs.append(item['Row'])
        idx.append(n)
    return Xs, Ys, idx

def create_grid():
   
    x_label = np.linspace(1, 64, 64)
    y_label = np.linspace(1, 64, 64)
    xx, yy = np.meshgrid(x_label, y_label, sparse=False, indexing='xy')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xx.flatten(), y=yy.flatten(), opacity = 0.5, marker={'color': 'green', 'showscale': False}, mode='markers'))
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
    fig2.update_layout(template="plotly_white", width=600, height=600, showlegend = False,legend=dict(orientation="h"),margin=dict(l=25,r=10,b=10,t=10,pad=4))
    return fig2

def create_sensor_grid(Xs,Ys):

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=Xs, y=Ys, opacity = 0.5, marker={'color': 'green', 'showscale': False}, mode='markers'))
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
    fig2.update_layout(template="plotly_white", width=600, height=600, showlegend = False,legend=dict(orientation="h"),margin=dict(l=25,r=10,b=10,t=10,pad=4))
    return fig2

def getChMap():

    newChs = np.zeros(4096, dtype=[('Row', '<i2'), ('Col', '<i2')])    
    idx = 0
    for idx in range(4096):
        column = (idx // 64) + 1
        row = idx % 64 + 1    
        if row == 0:
            row = 64
        if column == 0:
            column = 1

        newChs[idx] = (np.int16(row), np.int16(column))
        ind = np.lexsort((newChs['Col'], newChs['Row']))
    return newChs[ind]


class writeCBrw:
    def __init__(self, path, name,template,parameters):
        self.path = path
        self.fileName = name
        self.template = template
        self.description = parameters['Ver']
        self.version = parameters['Typ']
        self.brw = h5py.File(os.path.join(self.path, self.template), 'r')
        self.samplingrate = parameters['samplingRate']
        self.frames = parameters['nRecFrames']
        self.signalInversion = parameters['signalInversion']
        self.maxVolt = parameters['maxUVolt']
        self.minVolt = parameters['minUVolt']
        self.bitdepth = parameters['bitDepth']
        self.chs = parameters['recElectrodeList']
        self.QLevel = np.power(2, parameters['bitDepth'])
        self.fromQLevelToUVolt = (self.maxVolt - self.minVolt) / self.QLevel

    def createNewBrw(self):
        newName = os.path.join(self.path, self.fileName + '.brw')
        new = h5py.File(newName, 'w')

        new.attrs.__setitem__('Description', self.description)
        #new.attrs.__setitem__('GUID', self.brw.attrs['GUID'])
        new.attrs.__setitem__('Version', self.version)

        #new.copy(self.brw['3BRecInfo'], dest=new)
        #new.copy(self.brw['3BUserInfo'], dest=new)
        new.create_dataset('/3BRecInfo/3BRecVars/SamplingRate', data=[np.float64(100)])
        new.create_dataset('/3BRecInfo/3BRecVars/NewSampling', data=[np.float64(self.samplingrate)])
        new.create_dataset('/3BRecInfo/3BRecVars/NRecFrames', data=[np.float64(self.frames)])
        new.create_dataset('/3BRecInfo/3BRecVars/SignalInversion', data=[np.float64(self.signalInversion)])
        new.create_dataset('/3BRecInfo/3BRecVars/MaxVolt', data=[np.float64(self.maxVolt)])
        new.create_dataset('/3BRecInfo/3BRecVars/MinVolt', data=[np.float64(self.minVolt)])
        new.create_dataset('/3BRecInfo/3BRecVars/BitDepth', data=[np.float64(self.bitdepth)])
        new.create_dataset('/3BRecInfo/3BMeaStreams/Raw/Chs', data=[self.chs])
        new.create_dataset('/3BRecInfo/3BRecVars/Ver', data=[self.description])
        new.create_dataset('/3BRecInfo/3BRecVars/Typ', data=[self.version])

        self.newDataset = new
        self.newDataset.close()
       
    def appendBrw(self, fName, frames, chs,fs,NewSampling,ss,st):

        brwAppend = h5py.File(fName, 'a')

        signalInversion = brwAppend['3BRecInfo/3BRecVars/SignalInversion']
        maxVolt = brwAppend['3BRecInfo/3BRecVars/MaxVolt'][0]
        minVolt = brwAppend['3BRecInfo/3BRecVars/MinVolt'][0]
        QLevel = np.power(2, brwAppend['3BRecInfo/3BRecVars/BitDepth'][0])
        fromQLevelToUVolt = (maxVolt - minVolt) / QLevel

        del brwAppend['/3BRecInfo/3BRecVars/NewSampling']
        
        try:
            del brwAppend['/3BRecInfo/3BMeaStreams/Raw/Chs']
        except:
            del brwAppend['/3BRecInfo/3BMeaStreams/WaveletCoefficients/Chs']
            
        del brwAppend['/3BRecInfo/3BRecVars/NRecFrames']
        del brwAppend['/3BRecInfo/3BRecVars/SamplingRate']

        brwAppend.create_dataset('/3BRecInfo/3BMeaStreams/Raw/Chs', data=chs)
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/NRecFrames', data=[np.int64(frames)])
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/SamplingRate', data=[np.float64(fs)])
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/NewSampling', data=[np.float64(NewSampling)])
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/startTime', data=[np.float64(ss)])
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/endTime', data=[np.float64(st)])

        brwAppend.close()

    def close(self):
        self.newDataset.close()
        self.brw.close()

fig4 = go.Figure({ "layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False},
                "annotations": [{"text": "Select Groups & Generate Raster","xref": "paper",
                "yref": "paper","showarrow": False,"font": {"size": 22}}]}})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

fig0 = fig4
fig2 = create_grid()
color = '#B22222'
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': color,
    'color': '#F8F8FF',
    'padding': '6px'
}

button_style = {#'display': 'flex', 
            'flex-direction': 'column',
            'align-items': 'center',
            'padding': '6px 14px',
            'font-family': '-apple-system, BlinkMacSystemFont, "Roboto", sans-serif',
            'border-radius': '6px',
            'color': '#3D3D3D',
            'background':'#fff',
            'border': 'none',
            'box-shadow': '0px 0.5px 1px rgba(0, 0, 0, 0.1)',
            'user-select': 'none',
            'touch-action': 'manipulation',
            #'box-shadow': '0px 0.5px 1px rgba(0, 0, 0, 0.1), 0px 0px 0px 3.5px rgba(58, 108, 217, 0.5)',
            'outline': '0',
            }

table_dict0 = [{'File-Path': '', 'File-Name': '', 'Total-Active-Channels': 0, 'Frames': 0, 'Recording-Length': 0,
                'Sampling-Rate': 0}, ]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div(children=[html.Div([html.H2("Select, Downsample and Export: Channel Selection Toolbox")],
                                    style={'text-align': 'center', 'vertical-align': 'center',
                                                'border': 'thin black solid', 'background-color': color,
                                                'color': '#F8F8FF'}),
                                html.Div([

                                    html.Div([html.H5("Input File Path: ")],
                                             style=dict(width='15%', display='table-cell',paddingLeft='5%')),
                                    html.Div([dcc.Input(id='File-Name', type='text', size='50', debounce=True,placeholder="Measurement File & Path in \'.brw\' format ")],
                                             style=dict(width='35%', display='table-cell')),
                                   
                                    # html.Div([html.H6("Select Image File: ")],
                                    #          style=dict(width='10%', display='table-cell')),
                                    html.Div([dcc.Upload(id='upload-image',children= html.Button("Upload Slice Image File (*_cropped.jpg)",style=tab_selected_style),multiple=False), ],
                                             style=dict(width='50%', display='table-cell', padding='1%')), ],
                                    style=dict(width='100%', display='table'), ),

                                html.Div([dash_table.DataTable(id='table', columns=[
                                            {'name': 'File Path', 'id': 'File-Path', 'type': 'text'},
                                            {'name': 'File Name', 'id': 'File-Name', 'type': 'text'},
                                            {'name': 'Version', 'id': 'Ver', 'type': 'text'},
                                            {'name': 'Data Format', 'id': 'Typ', 'type': 'text'},
                                            {'name': 'Active Channels', 'id': 'Total-Active-Channels', 'type': 'numeric'},
                                            {'name': 'Data per Channel', 'id': 'Frames', 'type': 'numeric'},
                                            {'name': 'Recording Time (Seconds)', 'id': 'Recording-Length',
                                            'type': 'numeric'},
                                            {'name': 'Sampling (Hz)', 'id': 'Sampling-Rate', 'type': 'numeric'}],
                                            data=table_dict0, style_as_list_view=True,
                                            style_data={'border': '1px white', 'font_size': '16px','text_align': 'center'},
                                            style_header={'backgroundColor': 'white', 'border': '1px white', 'font_size': '18px','fontWeight': 'bold', 'text-align': 'center'}), ],
                                    style={'width': '100%'}),

                                html.Div(id='file_name_text', children='Analysis File: ',
                                    style={'text-align': 'center', 'vertical-align': 'bottom', 'border': 'thin black solid', 'width': '100%', 'background-color': '#A52A2A', 'display': 'None'}),

                                html.Hr(style={'width': '10%', 'border-top': '3px dotted black', 'border-radius': '5px'}),

                                html.Div([
                                    html.Div(children=[html.Div([html.H3('Select Channels for Export')],
                                            style={'text-align': 'center', 'width': '100%','padding-left': '12%'}),
                                        html.Div([dcc.Graph(id='input-grid', figure=fig2)],style={'text-align': 'center', 'width': '100%', 'padding-left': '5%'}),], 
                                            style={'text-align': 'center'}, className="four columns"),
                                        
                                    html.Div([
                                            html.Div([daq.LEDDisplay(id='chCount',label={'label': 'Channel Count','style': 
                                            {'backgroundColor': 'white','border': '5px white', 'font_size': '50px','fontWeight': 'bold','text-align': 'center'}},
                                            labelPosition='bottom',value='0',size=25,color="black",backgroundColor='white')],style={'padding-left':'0%','padding-top':'25%','padding-bottom':'25%'}),
                                            html.Div([
                                                html.Div([html.H5('# Rows to Skip:')],style={'text-align': 'left', 'width': '50%','display': 'table-cell'}), 
                                                html.Div([dcc.Input(id='row-step',type='number',value=0,step=1,min=0,max=3)],style={'text-align': 'right', 'width': '50%','display': 'table-cell'}),
                                            ],style=dict(width='110%', display='table')),
                                            html.Div([
                                                html.Div([html.H5('# Columns to Skip:')],style={'text-align': 'left', 'width': '50%','display': 'table-cell'}), 
                                                html.Div([dcc.Input(id='column-step',type='number',value=0,step=1,min=0,max=3)],style={'text-align': 'right', 'width': '50%','display': 'table-cell'}),
                                            ],style=dict(width='110%', display='table')),

                                            html.Div([
                                                html.Div([html.H5('Downsampling (Hz):')],style={'text-align': 'left', 'width': '90%','display': 'table-cell'}), 
                                                html.Div([dcc.Input(id='NewSampling',type='number',value=100,max=20000)],style={'text-align': 'right', 'width': '10%','display': 'table-cell'}),
                                            ],style=dict(width='110%', display='table')),

                                            html.Div([
                                                html.Div([html.H5('Start Time (s):')],style={'text-align': 'left', 'width': '90%','display': 'table-cell'}), 
                                                html.Div([dcc.Input(id='ss',type='number',value=0,max=50)],style={'text-align': 'right', 'width': '10%','display': 'table-cell'}),
                                            ],style=dict(width='110%', display='table')),

                                            html.Div([
                                                html.Div([html.H5('End Time (s):')],style={'text-align': 'left', 'width': '90%','display': 'table-cell'}), 
                                                html.Div([dcc.Input(id='st',type='number',)],style={'text-align': 'right', 'width': '10%','display': 'table-cell'}),
                                            ],style=dict(width='110%', display='table')),

                                        
                                            html.Div([ html.Button("Export Channels to *.brw File",id='gen-brw',style=tab_selected_style)], style={'text-align': 'center','padding':'12%'}),

                                        ], style={'text-align': 'center'},className='three columns'),
                                    
                                    html.Div(children=[html.Div([html.H3('Channels Exported')],
                                            style={'text-align': 'center', 'width': '100%','padding-left': '5%'}),
                                        html.Div([dcc.Graph(id='output-grid', figure=fig2)],style={'text-align': 'center', 'width': '100%', 'padding-left': '5%'}),], 
                                            style={'text-align': 'center'}, className="four columns"),
                                ],style={'text-align': 'center'}, className="row")

])


@app.callback(
    [Output('file_name_text', 'children'), Output('input-grid', 'figure'), Output('table', 'data'),Output('st','value')],
    Input('upload-image', 'contents'),State('File-Name', 'value'))
def update_grid(img, value):
    bool, parameters = check_filename(value)
    # print(value,bool)
    if value and bool == True:
        
        filepath = value
        h5 = h5py.File(filepath, 'r')
        
#        parameters = parameter(h5,)
        chsList = parameters['recElectrodeList']
        Frames = parameters['nRecFrames']
        endTime = parameters['recordingLength']
        file_name = filepath.split('\\')[-1]
        file_path = '\\'.join(filepath.split('\\')[0:-1])

    

        info_dict = {'Filename': str(filepath), 'Channels': str(len(chsList)),'Ver':parameters['Ver'], 'Typ':parameters['Typ'], 'Frames': str(parameters['nRecFrames']),
                     'Recording-Length': str(round(parameters['nRecFrames'] / parameters['samplingRate'])),
                     'Sampling': str(parameters['samplingRate'])}
        table_dict = [{'File-Path': str(file_path), 'File-Name': str(file_name), 'Ver':parameters['Ver'],'Typ':parameters['Typ'],'Total-Active-Channels': len(chsList),
                       'Frames': parameters['nRecFrames'],
                       'Recording-Length': round(parameters['nRecFrames'] / parameters['samplingRate']),
                       'Sampling-Rate': parameters['samplingRate']}, ]
        Xs, Ys, idx = get_chMap(chsList)
        fig2 = create_sensor_grid(Xs, Ys)
        fig2.update_layout(template="plotly_white", showlegend=False, clickmode='event+select', width=600, height=600,legend=dict(orientation="h"))
        fig2.update_layout(images=[dict(source=img,xref="paper",yref="paper",x=0.008,y=0.955,sizex=0.982,sizey=0.90,sizing="stretch",opacity=1,layer="below")],)
        fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', fixedrange=True,range=[0, 65], mirror=True)
        fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, fixedrange=True,range=[0, 65], autorange="reversed")
        h5.close()
    else:
        path = "Enter Complete File-Path and Filename to Begin [Example: C:/Users/amahadevan/Documents/slice1_resample.brw]"
        info_dict = {'Message': path}
        table_dict = [{'File-Path': '', 'File-Name': '',  'Ver':'NA','Typ':'NA','Total-Active-Channels': 0, 'Frames': 0, 'Recording-Length': 0,
                       'Sampling-Rate': 0}, ]
        fig2 = create_grid()
        endTime = 0

    return html.H6(json.dumps(info_dict)), fig2, table_dict,endTime

@app.callback(Output('chCount','value'), [Input('input-grid', 'selectedData'),Input('row-step','value'),Input('column-step','value')])
def channelCount(selectedData,row_step,column_step):
    points = selectedData

    if points is None:
        return 0
    else:
        chX = []
        chY = []
        for item in points["points"]:
            if item['y'] % (row_step+1) == 0 and item['x'] % (column_step+1)==0:
                chX.append(item['x'])
                chY.append(item['y'])
        
        default = len(chX)
        return default


@app.callback(Output('output-grid','figure'), Input('gen-brw','n_clicks'),
[State('file_name_text', 'children'),State('input-grid', 'selectedData'),State('row-step','value'),State('column-step','value'),State('upload-image', 'contents'),State('NewSampling','value'),State('ss','value'),State('st','value')])
def generate_brw(n_clicks,value,selectedData,row_step,column_step,img,NewSampling,ss,st):
    points = selectedData
    
    if n_clicks and points is not None:
        #print(value)
        path0 = value['props']
        path0 = json.loads(path0['children'])
        #print(path0)
        chX = []
        chY = []
        for item in points["points"]:
            if item['y'] % (row_step+1) == 0 and item['x'] % (column_step+1)==0:
                chX.append(item['x'])
                chY.append(item['y'])
        h5 = h5py.File(path0['Filename'], 'r')
        parameters = parameter(h5,path0['Ver'].lower())
        chsList = parameters['recElectrodeList']
        xs, ys, idx = get_chMap(chsList)
        h5.close()
        fig2 = get_grid(chX,chY,xs,ys)
        #fig2.update_layout(template="plotly_white", showlegend=True, clickmode='event+select', width=600, height=600,legend=dict(orientation="h"))
        fig2.update_layout(images=[dict(source=img,xref="paper",yref="paper",x=0.008,y=0.955,sizex=0.982,sizey=0.90,sizing="stretch",opacity=1,layer="below")],)
        fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', fixedrange=True,range=[0, 65], mirror=True)
        fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, fixedrange=True,range=[0, 65], autorange="reversed")

        newChs = np.zeros(len(chX), dtype=[('Row', '<i2'), ('Col', '<i2')])
        idx = 0
        for chX,chY in zip(chX,chY):
            newChs[idx] = (np.int16(chY), np.int16(chX)) # (row,col)
            idx += 1

        ind = np.lexsort((newChs['Col'], newChs['Row']))
        newChs = newChs[ind]
        #print(newChs)

        Input_file_path = path0['Filename']
        input_file_name = Input_file_path.split('\\')[-1]
        input_file_path = '\\'.join(Input_file_path.split('\\')[0:-1])

        output_file_name = input_file_name.split('.')[0]+"_exportCh"
        output_file_name_brw = input_file_name.split('.')[0]+"_exportCh"+".brw"
        output_path = input_file_path+'\\'+output_file_name_brw

        dset = writeCBrw(input_file_path, output_file_name,input_file_name,parameters)

        dset.createNewBrw()
        dset.appendBrw(output_path, path0['Frames'], newChs,path0['Sampling'],NewSampling,ss,st)
        
    else:
        fig2 = fig4

    return fig2

if __name__ == '__main__':
    app.run_server(port = '9090',debug=True)