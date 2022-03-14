import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.graph_objects as go
import dash_daq as daq
from dash import dash_table
import base64
from flask_caching import Cache
import json
import scipy
from scipy import signal
from typing import Optional
import numpy as np
import h5py
import pandas as pd
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import os
import scipy.io
import skimage.measure
import time
from datetime import datetime
import csv

def parameter(h5):
    parameters = {}
    parameters['nRecFrames'] = h5['/3BRecInfo/3BRecVars/NRecFrames'][0]
    parameters['samplingRate'] = h5['/3BRecInfo/3BRecVars/SamplingRate'][0]
    parameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
    parameters['signalInversion'] = h5['/3BRecInfo/3BRecVars/SignalInversion'][0]  # depending on the acq version it can be 1 or -1
    parameters['maxUVolt'] = h5['/3BRecInfo/3BRecVars/MaxVolt'][0]  # in uVolt
    parameters['minUVolt'] = h5['/3BRecInfo/3BRecVars/MinVolt'][0]  # in uVolt
    parameters['bitDepth'] = h5['/3BRecInfo/3BRecVars/BitDepth'][0]  # number of used bit of the 2 byte coding
    parameters['qLevel'] = 2 ^ parameters['bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
    parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
    parameters['recElectrodeList'] = list(h5['/3BRecInfo/3BMeaStreams/Raw/Chs'])  # list of the recorded channels
    parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])
    return parameters

def get_ch_number(x, y):
    ch_number = x * 64 + y % 64
    return ch_number

def get_row_col_num(ch_number):
    row = ch_number // 64
    column = ch_number % 64
    if column == 0:
        column = 64

    return row, column

def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = scipy.signal.butter(order, high, btype='highpass')
    return b, a

def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = scipy.signal.butter(order, low, btype='lowpass')
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

def cheby_highpass(cutoff, fs, order=6, ripple=2):
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = scipy.signal.cheby1(order, ripple, high, btype='highpass')
    return b, a

def cheby_lowpass(cutoff, fs, order=6, ripple=2):
    nyq = 0.5 * fs
    low = cutoff / nyq
    b, a = scipy.signal.cheby1(order, ripple, low, btype='lowpass')
    return b, a

def cheby_bandpass(lowcut, highcut, fs, order=6, ripple=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.cheby1(order, ripple, [low, high], btype='bandpass')
    return b, a

def frequency_filter(signal: np.ndarray, fs: int, type, lowcut: Optional[int], highcut: Optional[int],
                     order=6) -> np.ndarray:
    """
    Custom bandpass filter
    :param signal: single-channel signal
    :param fs: sampling rate [Hz]
    :param lowcut: cut everything below this frequency [Hz]
    :param highcut: cut everything above this frequency [Hz]
    :param order: order of the Butterworth filter
    :return: flitered signal (ndarray float32)
    """
    if lowcut == 0:
        lowcut = None
    if highcut == fs // 2:
        highcut = None

    if lowcut and highcut:
        if type == 'BTR':
            b, a = butter_bandpass(lowcut, highcut, fs, order)
        elif type == 'CBY':
            b, a = cheby_bandpass(lowcut, highcut, fs, order, 2)
    elif lowcut and not highcut:
        if type == 'BTR':
            b, a = butter_highpass(lowcut, fs, order)
        elif type == 'CBY':
            b, a = cheby_highpass(lowcut, fs, order, 2)
    elif highcut and not lowcut:
        if type == 'BTR' or type == 'ANDY':
            b, a = butter_lowpass(highcut, fs, order)
        elif type == 'CBY':
            b, a = cheby_lowpass(highcut, fs, order, 2)
    else:
        return signal

    y = scipy.signal.filtfilt(b, a, signal)
    return y

def fft(signal, fs):
    ts = 1 / fs
    n = signal.size
    freq = scipy.fft.fftfreq(n, d=ts)
    fft_signal = 2 * abs(scipy.fft.fft(signal)) / n
    subset = np.where(freq > 1)
    return freq[subset], fft_signal[subset]

def create_sensor_grid():
    grid = np.zeros(shape=(64, 64))
    x_label = np.linspace(1, 64, 64)
    y_label = np.linspace(1, 64, 64)
    xx, yy = np.meshgrid(x_label, y_label, sparse=False, indexing='xy')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xx.flatten(), y=yy.flatten(), marker={'color': 'red', 'showscale': False}, mode='markers',
                   name='InActive Channels'))
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
    fig2.update_layout(template="plotly_white", width=600, height=600, legend=dict(orientation="h"))
    return fig2

def get_channel_list(chsList):
    column_list = []
    for item in chsList:
        ch_number = get_ch_number(item['Col'], item['Row'])
        column_list.append(ch_number)

    return column_list

def get_reference_index(xx_sig,sampling):
        ss = 0
        ee = ss + 60

        for i in range(int(len(xx_sig)/sampling)):
            ss_i =i
            ee_i = ss_i + 60
            xmean = np.mean(xx_sig[int(sampling * ss):int(sampling * ee)])
            xstd = np.std(xx_sig[int(sampling * ss):int(sampling * ee)])
            xmean_i = np.mean(xx_sig[int(sampling * ss_i):int(sampling * ee_i)])
            xstd_i = np.std(xx_sig[int(sampling * ss_i):int(sampling * ee_i)])

            v = xmean+xstd
            v_i = xmean_i+xstd_i

            if v > v_i:

                ss = ss_i
                ee = ss+60
                
        return ss+60, ee+60

def get_row_column_list(data, chsList, parameters):
    data = convert_to_uV(data, parameters) / 1000000
    sampling = parameters['samplingRate']
    y_active = []
    x_active = []
    y_noise = []
    x_noise = []
    count = 0
    for item in chsList:
        row = item['Row']
        column = item['Col']
        x = data[:, count][0:int(10 * 60 * sampling)]
        x = x - np.mean(x)
        sig = abs(x)
        peaks, properties = find_peaks(sig, prominence=float(0.07), width=(0.02 * float(sampling)))
        if len(peaks) > 10:
            y_noise.append(row)
            x_noise.append(column)
        else:
            y_active.append(item['Row'])
            x_active.append(item['Col'])
        count += 1
    return y_active, x_active, y_noise, x_noise

def convert_to_uV(data, parameters):
    ADCCountsToMV, MVOffset = Digital_to_Analog(parameters)
    data = data * ADCCountsToMV + MVOffset
    return data

def Digital_to_Analog(parameters):
    ADCCountsToMV = parameters['signalInversion'] * parameters['fromQLevelToUVolt']
    MVOffset = parameters['signalInversion'] * parameters['minUVolt']
    return ADCCountsToMV, MVOffset


def peak_raster_grid(data, column_list, parameters, prom, wid, detect_mode, frame_value):
    data = convert_to_uV(data, parameters) / 1000000
    df = pd.DataFrame(data, columns=column_list)
    fig3 = go.Figure()
    fig9 = go.Figure()
    lfp = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
    pp_prominence = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
    pp_width = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
    range_value0 = int(frame_value[0] / parameters['samplingRate'])
    range_value1 = int(frame_value[1] / parameters['samplingRate'])

    for i in column_list:
        lfp_sub = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
        pp_prominence_sub = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
        pp_width_sub = np.zeros(parameters['nRecFrames'])[frame_value[0]:frame_value[1]]
        x = df[i].to_numpy()
        x = x - np.mean(x)
        spikes = np.arange(0, parameters['nRecFrames'], 1)
        if detect_mode == 'DS':
            sig = abs(x)
        else:
            sig = x
        peaks, properties = find_peaks(sig, prominence=prom, width=wid)
        lfp_sub[peaks] = 1
        pp_prominence_sub[peaks] = abs(sig)[peaks]

        pp_width_sub[peaks] = properties['widths'] / parameters['samplingRate']
        lfp += lfp_sub
        pp_prominence += pp_prominence_sub
        pp_width += pp_width_sub
        row, column = get_row_col_num(i)
        spikes[peaks] = i


        x = np.arange(0, parameters['nRecFrames'], 1) / parameters['samplingRate']
        x = x[frame_value[0]:frame_value[1]]
        y = str(row) + ', ' + str(column)
        fig3.add_trace(go.Scatter(x=x[peaks], y=spikes[peaks], mode='markers', marker_color='black', marker_size=2, name=y))
        fig3.update_layout(template="plotly_white", showlegend=False, width=600, height=600)
        fig3.update_xaxes(showline=True, linewidth=1, showgrid=False, linecolor='black', mirror=True)
        fig3.update_yaxes(showline=True, linewidth=1, showgrid=False, linecolor='black', mirror=True)
 
    mean_LFP = np.sum(lfp) / (range_value1 - range_value0)
    mean_amplitude = np.sum(pp_prominence) / (range_value1 - range_value0)
    mean_width = np.sum(pp_width) / (range_value1 - range_value0)

    time = np.arange(0, parameters['nRecFrames'], 1) / parameters['samplingRate']
    xx = np.arange(0, parameters['nRecFrames'], 1) / parameters['samplingRate']
    time_range0 = xx[frame_value[0]]
    time_range1 = xx[frame_value[1]]

    axis_template = dict(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,
                         mirror=True)
    fig9 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=('Count', 'Amplitude', 'Duration'))
    fig9.update_layout(template="plotly_white", xaxis=axis_template, yaxis=axis_template, showlegend=False, width=600,
                       height=500, title_text='LFP Events')
    fig9.add_trace(go.Scatter(x=time[frame_value[0]:frame_value[1]], y=lfp), row=1, col=1)
    fig9.add_trace(go.Scatter(x=time[frame_value[0]:frame_value[1]], y=pp_prominence), row=2, col=1)
    fig9.add_trace(go.Scatter(x=time[frame_value[0]:frame_value[1]], y=pp_width), row=3, col=1)
    fig9.update_xaxes(showgrid=False, linecolor='black', showticklabels=False, linewidth=2, showline=True,
                      mirror=True, row=1, col=1)
    fig9.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,
                      mirror=True, row=1, col=1)
    fig9.update_xaxes(showgrid=False, linecolor='black', showticklabels=False, linewidth=2, showline=True,
                      mirror=True, row=2, col=1)
    fig9.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,
                      mirror=True, row=2, col=1)
    fig9.update_xaxes(title_text='Time, Seconds', showgrid=False, linecolor='black', showticklabels=True, linewidth=2,
                      showline=True,
                      mirror=True, row=3, col=1)
    fig9.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,
                      mirror=True, row=3, col=1)


    fig3.update_xaxes(title_text='Time, Seconds', range=[time_range0, time_range1])
    fig3.update_yaxes(title_text='Channel Number', range=[1, 4096])
    axis_template = dict(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,
                    mirror=True)
    fig3.update_layout(template="plotly_white", xaxis=axis_template, yaxis=axis_template,
                    showlegend=False, width=600, height=600, legend=dict(traceorder="normal",
                    font=dict(family="sans-serif",size=12,color="black")))

    return fig3, fig9, mean_LFP, mean_amplitude, mean_width

def check_filename(path):
    try:
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = parameters['recElectrodeList']
        Frames = parameters['nRecFrames']
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chsList))
        return True
    except:
        return False

def range_slider_marks(min, max, steps):
    marks = {}
    steps_array = np.linspace(min, max, steps)
    for i in steps_array:
        marks[i] = str(round((i), 0))
    marks[0] = str(0) + ' minutes'
    marks[max] = str(round((max), 0)) + ' seconds'

    return marks

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'full') / w

def psd(x, fs):
    f, t, Zxx = signal.stft(x, fs, nperseg=fs, noverlap=0, window='hann')
    spec = 2 * abs(Zxx) / (len(x) / fs)
    power = np.sqrt(np.sum(np.power(abs(Zxx), 2), axis=0))
    return f, t, power, spec


def get_events_envelope(data, sampling, Frames, detect_mode, smooth1, cutoff1, smooth2, cutoff2,psd_threshold,psd_window):
    x = data
    sig = data.copy()
    xx_sig = sig[int(sampling * 60):int(sampling * 300)]   
    ss, ee = get_reference_index(xx_sig,sampling)
    x_sig = sig[int(sampling *ss):int(sampling * ee)]    
    threshold = 6 * np.std(x_sig)
    f_ref, t_ref, Zxx_ref, spec_ref = psd(x_sig, sampling)
    power_spec_ref = np.mean(Zxx_ref) + 6*np.std(Zxx_ref)

    ff, tt, Zxx, spec = psd(x, sampling)
    power_spec = Zxx
    freq_spec = np.zeros((len(tt), 1))
    freq_idx = np.where(power_spec >= power_spec_ref)
    freq_spec[freq_idx] = 1

    wid = 0.035
    wid = float(wid) * float(sampling)
    peaks_raster = np.zeros(len(sig))

    if detect_mode == 'DS':
        sig_peak_find = abs(sig)
    else:
        sig_peak_find = sig

    peaks, properties = find_peaks(sig_peak_find, prominence=threshold, width=wid)
    peaks_raster[peaks] = 1

    t = np.linspace(0, Frames, Frames) / sampling
    mov_avg = moving_average(abs(np.diff(peaks_raster)), smooth1)
    idx = np.where(mov_avg >= cutoff1)
    index_tt = len(mov_avg) - len(t)
    mov_avg = np.zeros(len(mov_avg))
    mov_avg[idx] = 1
    label = skimage.measure.label(mov_avg, connectivity=1)

    # Conv-2 for Peak Find
    CID = moving_average(abs(np.diff(label[0:len(t)])), smooth2)
    idx = np.where(CID >= cutoff2)
    CID = np.zeros(len(CID))
    index_tt = len(CID) - len(t)
    CID[idx] = 1
    label2 = skimage.measure.label(CID, connectivity=1)

    index_t = len(label2) - len(t)
    events_id = np.where(label2[0:len(t)] > 0)
    events = np.zeros(Frames, dtype=int)
    events[events_id] = 1

    # CONV-2 and connected components
    mov_avg_pp = moving_average(abs(np.diff(freq_spec[:, 0])), 25)
    idx_pp = np.where(mov_avg_pp >= 0.001)
    index_tt_pp = len(mov_avg_pp) - len(tt)
    mov_avg_pp = np.zeros(len(mov_avg_pp))
    mov_avg_pp[idx_pp] = 1
    label_pp = skimage.measure.label(mov_avg_pp, connectivity=1)

    events_pp_id = np.where(label_pp > 0)
    events_pp = np.zeros((len(mov_avg_pp)), dtype=int)
    events_pp[events_pp_id[0]] = 1

    return t, events, tt, events_pp[index_tt_pp:], peaks_raster


def get_seizures_df(df, idd):
    df_min = df.groupby(by=[idd], as_index=False)['time'].agg(min).rename(columns={idd: idd, 'time': 'start'})
    df_max = df.groupby(by=[idd], as_index=False)['time'].agg(max).rename(columns={idd: idd, 'time': 'end'})
    df_seizures = pd.DataFrame(columns=[idd])
    df_seizures[idd] = df[idd].unique()
    df_seizures = df_seizures.merge(df_min, on=idd, how='left')
    df_seizures = df_seizures.merge(df_max, on=idd, how='left')
    df_seizures['duration'] = df_seizures['end'] - df_seizures['start']
    df_seizures['tbe'] = df_seizures['start'] - df_seizures['end'].shift(1)
    df_seizures.loc[0, 'tbe'] = 0
    return df_seizures


def classify_events(X):
    type_dict = {'interictal': 'IC', 'seizure': 'SZ', }
    typ = ''
    if X >= 3 and X <= 10:
        typ = type_dict['interictal']
    elif X > 10:
        typ = type_dict['seizure']
    elif X < 3:
        typ = 'NS'

    return typ


def combine_events(df_seizures):
    df_events = pd.DataFrame(columns=['id', 'start', 'end'])
    events_dict = {}
    row_count = 0
    df_compare = df_seizures.copy()

    sz_index = list(df_compare.index)
    seizure_number = list(df_compare['id-ps'])
    seizure_start = list(df_compare['start'])
    seizure_end = list(df_compare['end'])
    dur_iter = list(df_compare['duration'])
    time_before_events = list(df_compare['tbe'])

    count = 0

    for i in range(len(sz_index)):
        events = {}

        if time_before_events[i] < 18 and i > 0:
            df_events.loc[count - 1, 'end'] = seizure_end[i]

        else:
            events['id'] = count
            events['start'] = seizure_start[i]
            events['end'] = seizure_end[i]
            df_events = df_events.append(events, ignore_index=True)
            count += 1

    df_events['duration'] = df_events['end'] - df_events['start']
    df_events['tbe'] = df_events['start'] - df_events['end'].shift(1)
    df_events.loc[0, 'tbe'] = 0
    df_events['type'] = 'SZ'
    return df_events


def detect_seizures(events, events_pp, tt, t):

    lfp_peaks = np.zeros((len(events), 3), dtype=int)
    event_label = skimage.measure.label(events, connectivity=1)

    lfp_peaks[:, 0] = t
    lfp_peaks[:, 2] = events
    lfp_peaks[:, 1] = event_label

    spectrum_peaks = np.zeros((len(tt), 3), dtype=int)
    label_spectrum_peaks = skimage.measure.label(events_pp, connectivity=1)
    spectrum_peaks[:, 0] = tt
    spectrum_peaks[:, 1] = label_spectrum_peaks
    spectrum_peaks[:, 2] = events_pp

    df_peaks = pd.DataFrame(lfp_peaks, columns=['time', 'id-pp', 'bool_pp'])
    df_peaks['time'] = df_peaks['time'].astype(int)
    df_peaks = df_peaks.groupby(by=['time'], as_index=False).agg(max)

    df_spectrum_peaks = pd.DataFrame(spectrum_peaks, columns=['time', 'id-ps', 'bool_ps'])
    df_spectrum_peaks['time'] = df_spectrum_peaks['time'].astype(int)
    
    df_peaks = df_peaks.merge(df_spectrum_peaks, on='time', how='left')
    df_peaks['bool_match'] = np.where((df_peaks['bool_pp'] == 1) & (df_peaks['bool_ps'] == 1), 1, 0)
    label_pp_ps = skimage.measure.label(df_peaks['bool_match'], connectivity=1)
    df_peaks['label'] = label_pp_ps
    df_duration = (df_peaks.groupby(by=['label'], as_index=False)['bool_match'].agg(sum)).rename(columns={'label': 'label', 'bool_match': 'duration'})
    df_peaks = df_peaks.merge(df_duration, on='label', how='left')
    df_peaks['bool_match_1'] = np.where((df_peaks['duration'] > 5), 1, 0)
    df_peaks['label2'] = np.where(((df_peaks['bool_match_1'] == 1) & (df_peaks['label'] > 0)), 1, 0)
    
    df = df_peaks[['time', 'id-pp', 'id-ps', 'bool_pp', 'bool_ps', 'label', 'label2']].copy()
    df_peaks = df_peaks[df_peaks['label2'] == 1]
    df = df[df['id-ps'].isin(list(df_peaks['id-ps']))]
    df_seizures = get_seizures_df(df, 'id-ps')
    df_seizures['type'] = df_seizures['duration'].apply(classify_events)
    
    df_sz = combine_events(df_seizures)

    return df_sz


def get_psd(x, sampling):
    if len(x) < 1024:
        nperseg = len(x)
    else:
        nperseg = 1024

    f, pxx = signal.welch(x, sampling, 'hanning', nperseg, scaling='density')
    return f, pxx

def mean_sampling_frequency(t):
    n = len(t)
    if n>0:
        t_n = t[-1]
        t_1 = t[0]
        mean_sampling_frequency = (n - 1) / (t_n - t_1)
    else:
        mean_sampling_frequency = 0

    return mean_sampling_frequency


def mean_instantaneous_frequency(t):
    t_0 = t
    df = pd.DataFrame(t_0, columns=['t0'])
    df['t1'] = df['t0'].shift(1)
    df['delta_t'] = df['t0'] - df['t1']
    df['frequency'] = 1 / df['delta_t']
    df = df[df['frequency'] <= 1000]
    freq = np.array(df['frequency'])
    mean_instantaneous_frequency = np.mean(freq[1:])
    mean_inter_spike_interval = np.mean(np.array(df['delta_t'])[1:])

    return mean_instantaneous_frequency, mean_inter_spike_interval

def check_false_positive(df_loop,peaks_raster,sampling, Frames):
    sz_s = []
    sz_e = []
    sz_num = []
    num = 0
    for i in df_loop.index:
        if df_loop.loc[i, 'type'] != 'NS' and (~np.isnan(df_loop.loc[i,'start'])):
            ss = df_loop.loc[i, 'start']
            ee = df_loop.loc[i, 'end']
            sz_s.append(ss)
            sz_e.append(ee)
            sz_num.append(num)
            num = num+1

    IEF = get_ief(peaks_raster,sampling,Frames,sz_s,sz_e,sz_num)
    if IEF:
        df_record = pd.DataFrame.from_records(IEF)
        df_loop['peaks'] = df_record['count']
        df_loop = df_loop[df_loop['peaks']>10]
        
    return df_loop
        


def get_ief(peaks_raster, sampling, Frames, starts, ends, sz_num):
    time = np.arange(0, Frames, 1) / sampling
    ief = []
    for s, e, c in zip(starts, ends, sz_num):
        df_dict = {}
        t = time.copy()
        peaks = np.where(peaks_raster == 1)
        t = t[peaks]
        idx = np.where((t > s) & (t < e))
        t = t[idx]
        df_dict['sz_num'] = c+1
        df_dict['start'] = s
        df_dict['end'] = e
        df_dict['count'] = len(t)
        df_dict['msf'] = round(mean_sampling_frequency(t),3)
        m_isf, m_isi = mean_instantaneous_frequency(t)
        df_dict['m_isf'] = round(m_isf,3)
        df_dict['mean_isi'] = round(m_isi,2)
        ief.append(df_dict)

    return ief


def get_grid(column_list,row_list, column_20active,row_20active, selected_rows,selected_columns):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=column_list, y=row_list, marker={'color': 'grey', 'showscale': False}, mode='markers',
                       name='All Channels'))
    fig2.add_trace(go.Scatter(x=column_20active, y=row_20active, marker={'color': 'green', 'showscale': False}, marker_size = 7,
                       mode='markers',name='Active Channels'))
    fig2.add_trace(go.Scatter(x=column_20active[0:20], y=row_20active[0:20], marker={'color': 'red', 'showscale': False}, marker_size = 7,
                       mode='markers',name='20 Most Active Channels'))
    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
    fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65],autorange="reversed")
    fig2.update_layout(template="plotly_white", showlegend= True, width=700, height=700,legend=dict(orientation="h"))

    return fig2



fig = go.Figure(
    {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "Select a Channel to View",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 22
                    }
                }
            ]
        }
    })

fig5 = go.Figure(
    {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "Select a Time Segment to View FFT",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 22
                    }
                }
            ]
        }
    })

fig6 = go.Figure(
    {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "Updating Raster ...",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 22
                    }
                }
            ]
        }
    })
fig4 = go.Figure(
    {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": "Select Groups & Generate Raster",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 22
                    }
                }
            ]
        }
    })

# *** Dash Part ***
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

fig0 = fig
fig2 = create_sensor_grid()
fig3 = fig
fig1 = fig6
fig9 = fig

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
    'backgroundColor': '#483D8B',
    'color': '#F8F8FF',
    'padding': '6px'
}

table_dict0 = [{'File-Path': '', 'File-Name': '', 'Total-Active-Channels': 0, 'Frames': 0, 'Recording-Length': 0,
                'Sampling-Rate': 0}, ]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# cache = Cache(app.server, config={'CACHE_TYPE': 'redis', 'CACHE_REDIS_URL': os.environ.get('REDIS_URL', '')})
server = app.server
app.config.suppress_callback_exceptions = True

# timeout = 200

app.layout = html.Div(children=[html.Div([html.H1("XENON LFP Analysis Platform")],
                                         style={'text-align': 'center', 'vertical-align': 'bottom',
                                                'border': 'thin black solid', 'background-color': '#483D8B',
                                                'color': '#F8F8FF'}),
                                html.Div([
                                    html.Div([html.H6("Input File Path: ")],
                                             style=dict(width='12%', display='table-cell', padding='2%')),
                                    html.Div([dcc.Input(id='File-Name', type='text', size='85', debounce=True,
                                                        placeholder="Analysis File & Path in \'.brw\' format "),
                                              html.Button('Submit', id='submit-file', n_clicks=0)],
                                             style=dict(width='55%', display='table-cell')),
                                    html.Div([html.H6("Select Image File: ")],
                                             style=dict(width='10%', display='table-cell')),
                                    html.Div([dcc.Upload(id='upload-image',
                                                         children=html.Div(
                                                             ['Drag and Drop or ', html.A('Select File')]),
                                                         style={'width': '100%', 'height': '100%', 'lineHeight': '60px',
                                                                'borderWidth': '1px', 'borderStyle': 'dashed',
                                                                'borderRadius': '5px', 'textAlign': 'center',
                                                                'margin': '10px'},
                                                         multiple=False), ],
                                             style=dict(width='20%', display='table-cell', paddingRight='4%')), ],
                                    style=dict(width='100%', display='table'), ),
                                html.Div([
                                    dash_table.DataTable(id='table', columns=[
                                        {'name': 'File Path', 'id': 'File-Path', 'type': 'text'},
                                        {'name': 'File Name', 'id': 'File-Name', 'type': 'text'},
                                        {'name': 'Active Channels', 'id': 'Total-Active-Channels', 'type': 'numeric'},
                                        {'name': 'Data per Channel', 'id': 'Frames', 'type': 'numeric'},
                                        {'name': 'Recording Time (Seconds)', 'id': 'Recording-Length',
                                         'type': 'numeric'},
                                        {'name': 'Sampling (Hz)', 'id': 'Sampling-Rate', 'type': 'numeric'}],
                                                         data=table_dict0, style_as_list_view=True,
                                                         style_data={'border': '1px white', 'font_size': '16px',
                                                                     'text_align': 'center'},
                                                         style_header={'backgroundColor': 'white',
                                                                       'border': '1px white', 'font_size': '18px',
                                                                       'fontWeight': 'bold',
                                                                       'text-align': 'center'}), ],
                                    style={'width': '100%'}),
                                html.Div(id='file_name_text', children='Analysis File: ',
                                         style={'text-align': 'center', 'vertical-align': 'bottom',
                                                'border': 'thin black solid', 'width': '100%',
                                                'background-color': '#A52A2A', 'display': 'None'}),
                                html.Div(html.P([html.Br()])),
                                html.Div([
                                    html.Div([html.H6("Select a Time Range for Analysis: ")],
                                             style={'width': '20%', 'vertical-align': 'middle',
                                                    'display': 'table-cell'}),
                                    html.Div([dcc.RangeSlider(id='my-range-slider', min=0, max=20, value=[0, 10],
                                                              tooltip={'always_visible': True},
                                                              allowCross=False, dots=False)],
                                             style={'width': '78%', 'vertical-align': 'middle',
                                                    'display': 'table-cell'}),
                                ], style=dict(width='100%', display='table')),
                                html.Hr(
                                    style={'width': '10%', 'border-top': '3px dotted black', 'border-radius': '5px'}),
                                html.Div([
                                    html.Div(children=[
                                        html.Div([html.H5('Select Channels for Plots')],
                                                 style={'text-align': 'center', 'width': '100%'}),
                                        html.Div([dcc.Tabs([
                                            dcc.Tab(label='Active Channels', style=tab_style,
                                                    selected_style=tab_selected_style,
                                                    children=[dcc.Graph(id='g2', figure=fig2)], ),
                                            dcc.Tab(label='Channel Groups', style=tab_style,
                                                    selected_style=tab_selected_style, children=[
                                                    dcc.Tabs([
                                                        dcc.Tab(label='Group-1', style=tab_style,
                                                                selected_style=tab_selected_style,
                                                                children=[dcc.Graph(id='g2_g1', figure=fig2)]),
                                                        dcc.Tab(label='Group-2', style=tab_style,
                                                                selected_style=tab_selected_style,
                                                                children=[dcc.Graph(id='g2_g2', figure=fig2)]),
                                                        dcc.Tab(label='Group-3', style=tab_style,
                                                                selected_style=tab_selected_style,
                                                                children=[dcc.Graph(id='g2_g3', figure=fig2)]),
                                                    ]),
                                                ]), ]), ],
                                            style={'text-align': 'center', 'width': '100%', 'padding-left': '5%'}), ],
                                        style={'text-align': 'center'}, className="three columns"),

                                    html.Div([html.Br()], className='one columns'),
                                    html.Div([html.H5('LFP Raster Plots'),
                                              html.Div([dcc.Tabs([
                                                  dcc.Tab(label='LFP Detection (All Active Channels)',
                                                          style=tab_style, selected_style=tab_selected_style, children=[
                                                          html.Div([dcc.Tabs(vertical=False, children=[
                                                              dcc.Tab(label='Plot', style=tab_style,
                                                                      selected_style=tab_selected_style,
                                                                      children=[dcc.Graph(id='g3', figure=fig1)], ),
                                                              dcc.Tab(label='Summary', style=tab_style,
                                                                      selected_style=tab_selected_style,
                                                                      children=[dcc.Graph(id='g9', figure=fig1),
                                                                                html.Div([
                                                                                    html.Div([daq.LEDDisplay(id='lfp',
                                                                                                             label="LFP/S",
                                                                                                             value='0.0',
                                                                                                             size=18,
                                                                                                             color="black",
                                                                                                             backgroundColor="#48D1CC")],
                                                                                             style=dict(width='30%',
                                                                                                        display='table-cell')),
                                                                                    html.Div([daq.LEDDisplay(
                                                                                        id='amplitude',
                                                                                        label="Amplitude/S",
                                                                                        value='0.0', size=18,
                                                                                        color="black",
                                                                                        backgroundColor="#55B4B0")],
                                                                                        style=dict(width='30%',
                                                                                                   display='table-cell')),
                                                                                    html.Div([daq.LEDDisplay(
                                                                                        id='duration',
                                                                                        label="Duration/S", value='0.0',
                                                                                        size=18, color="black",
                                                                                        backgroundColor="#55B4B0")],
                                                                                        style=dict(width='30%',
                                                                                                   display='table-cell')), ],
                                                                                    style=dict(width='100%',
                                                                                               display='table'))]), ]), ], ), ], ),
                                                  dcc.Tab(label='Channel Raster (Groups)', style=tab_style,
                                                          selected_style=tab_selected_style, children=[
                                                          html.Div([dcc.Tabs(vertical=False, children=[
                                                              dcc.Tab(label='Plot', style=tab_style,
                                                                      selected_style=tab_selected_style,
                                                                      children=[dcc.Graph(id='g7', figure=fig3)], ),
                                                              dcc.Tab(label='Summary', style=tab_style,
                                                                      selected_style=tab_selected_style,
                                                                      children=[
                                                                      html.Br(),

                                                                     dcc.Tabs(vertical = True, children =[

                                                                        dcc.Tab(label='ALL Groups Summary', value = "all_g", style=tab_style, selected_style=tab_selected_style,
                                                                            children = [dcc.Graph(id='g9_ch', figure=fig9)]),
                                                                        dcc.Tab(label='Group-1 Channels', value = "g1",style=tab_style, selected_style=tab_selected_style,
                                                                            children = [dcc.Graph(id='g1_GRID', figure=fig9)]),
                                                                        dcc.Tab(label='Group-2 Channels', value = "g2", style=tab_style, selected_style=tab_selected_style,
                                                                            children = [dcc.Graph(id='g2_GRID', figure=fig9)]),
                                                                        dcc.Tab(label='Group-3 Channels', value = "g3", style=tab_style, selected_style=tab_selected_style,
                                                                            children = [dcc.Graph(id='g3_GRID', figure=fig9)]),
                                                                        ], value = "all_g", 
                                                                      style={'float': 'left', 'width': '100%'}),
                                                                                html.Div([
                                                                                    dash_table.DataTable(id='table2',
                                                                                                         columns=[
                                                                                                             {
                                                                                                                 'name': 'Group',
                                                                                                                 'id': 'Group',
                                                                                                                 'type': 'numeric'},
                                                                                                             {
                                                                                                                 'name': 'Total LFP Count (count/s)',
                                                                                                                 'id': 'LFP-Count',
                                                                                                                 'type': 'numeric'},
                                                                                                             {
                                                                                                                 'name': 'Total Channels',
                                                                                                                 'id': 'Tot-Channel',
                                                                                                                 'type': 'numeric'},
                                                                                                             {
                                                                                                                 'name': 'Active Channels',
                                                                                                                 'id': 'Act-Channel',
                                                                                                                 'type': 'numeric'},
                                                                                                             {
                                                                                                                 'name': 'LFP Count [Top 20] (count/s)',
                                                                                                                 'id': 'LFP-Count20',
                                                                                                                 'type': 'numeric'
                                                                                                             },
                                                                                                             # {
                                                                                                             #     'name': 'LFP/s',
                                                                                                             #     'id': 'LFP-Count-Time',
                                                                                                             #     'type': 'numeric'
                                                                                                             # },
                                                                                                             {
                                                                                                                 'name': 'Mean Amplitude [Top 20] (mV)',
                                                                                                                 'id': 'Mean-Amplitude',
                                                                                                                 'type': 'numeric'},
                                                                                                             {
                                                                                                                 'name': 'Mean Duration [Top 20] (S)',
                                                                                                                 'id': 'Mean-Duration',
                                                                                                                 'type': 'numeric'},

                                                                                                         ],
                                                                                                         data=[],
                                                                                                         style_as_list_view=True,
                                                                                                         style_data={
                                                                                                             'border': '1px white',
                                                                                                             'font_size': '12px',
                                                                                                             'text_align': 'center'},
                                                                                                         style_cell={
                                                                                                             'height': 'auto',
                                                                                                             # all three widths are needed
                                                                                                             'minWidth': '50px',
                                                                                                             'width': '100px',
                                                                                                             'maxWidth': '100px',
                                                                                                             'whiteSpace': 'normal'
                                                                                                         },
                                                                                                         style_header={
                                                                                                             'backgroundColor': 'white',
                                                                                                             'border': '1px white',
                                                                                                             'font_size': '14px',
                                                                                                             'fontWeight': 'bold',
                                                                                                             'text-align': 'center'})],
                                                                                    style=dict(width='100%',
                                                                                               display='table'))]), ]), ], ),
                                                      ]), ]), ], style={'float': 'left', 'width': '100%'}), ],
                                             style={'text-align': 'center'}, className="five columns"),
                                    html.Div([html.H5('Analysis Settings'),
                                              html.Div([dcc.Tabs([
                                                  dcc.Tab(label='LFP Detection', style=tab_style,
                                                          selected_style=tab_selected_style, children=[
                                                          html.H5('Digital Filter Parameters:'),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div(['Lower Cut-Off Frequency:  ', dcc.Input(id='lower',
                                                                                                            placeholder='Low cut-off Frequency:  ',
                                                                                                            type='number',
                                                                                                            value='0')]),
                                                          html.Div(['Upper Cut-Off Frequency:  ', dcc.Input(id='upper',
                                                                                                            placeholder='Upper cut-off Frequency',
                                                                                                            type='number',
                                                                                                            value='100')]),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div([
                                                              html.Div([daq.ToggleSwitch(id='my-toggle-switch',
                                                                                         label='Turn-ON-Filter',
                                                                                         labelPosition='top',
                                                                                         color='green', value=False, )],
                                                                       style=dict(width='40%', display='table-cell')),
                                                              html.Div([dcc.RadioItems(id='TYPE', options=[
                                                                  {'label': 'Butterworth', 'value': 'BTR'},
                                                                  {'label': 'Chebyshev', 'value': 'CBY'}, ],
                                                                                       value='BTR',
                                                                                       labelStyle={'display': 'block'},
                                                                                       inputStyle={
                                                                                           "margin-left": "10px",
                                                                                           "margin-right": "10px"}, ), ],
                                                                       style=dict(width='60%', display='table-cell',
                                                                                  align='top', )),
                                                          ], style=dict(width='100%', display='table')),

                                                          html.Div(html.P([html.Br()])),
                                                          html.H5('LFP Parameters:'),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div([
                                                              html.Div(['Threshold (mV):'],
                                                                       style={'text-align': 'left', 'width': '60%',
                                                                              'display': 'table-cell'}),
                                                              html.Div([dcc.Input(id='prominence',
                                                                                  placeholder='Threshold in (volts)',
                                                                                  type='number', value='0.07')],
                                                                       style={'text-align': 'right', 'width': '20%',
                                                                              'display': 'table-cell'}),
                                                          ], style=dict(width='100%', display='table')),
                                                          html.Div([
                                                              html.Div(['Time duration (Seconds):'],
                                                                       style={'text-align': 'left', 'width': '60%',
                                                                              'display': 'table-cell'}),
                                                              html.Div([dcc.Input(id='width',
                                                                                  placeholder='Peak Width in (seconds)',
                                                                                  type='number', value='0.02')],
                                                                       style={'text-align': 'right', 'width': '20%',
                                                                              'display': 'table-cell'}),
                                                          ], style=dict(width='100%', display='table')),
                                                          html.Div(html.P([html.Br()])),
                                                          dcc.RadioItems(id='detect_mode', options=[
                                                              {'label': 'Upward Peak Only', 'value': 'SS'},
                                                              {'label': 'Upward & Downward Peak', 'value': 'DS'}, ],
                                                                         value='DS', labelStyle={'display': 'block'},
                                                                         inputStyle={"margin-left": "20px",
                                                                                     "margin-right": "20px"},
                                                                         style={'text-align': 'justify'}),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div([html.Div([html.Button(
                                                              'Apply Settings and Generate Plot', id='button-2',
                                                              n_clicks=0,style=tab_selected_style)], style=dict(width='100%',
                                                                                       display='table-cell')), ],
                                                              style=dict(width='100%', display='table')),
                                                          # html.Div(html.P([html.Br()])),
                                                      ], ),
                                                  dcc.Tab(label='Channel Raster', style=tab_style,
                                                          selected_style=tab_selected_style, children=[
                                                          html.H5('Digital Filter Parameters:'),
                                                          html.Div(['Lower Cut-Off Frequency:  ',
                                                                    dcc.Input(id='lower_ras',
                                                                              placeholder='Low cut-off Frequency:  ',
                                                                              type='number', value='0')]),
                                                          html.Div(['Upper Cut-Off Frequency:  ',
                                                                    dcc.Input(id='upper_ras',
                                                                              placeholder='Upper cutt-off Frequency',
                                                                              type='number', value='100')]),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div([
                                                              html.Div([daq.ToggleSwitch(id='my-toggle-switch_ras',
                                                                                         label='Turn-ON-Filter (Raster)',
                                                                                         labelPosition='top',
                                                                                         color='green', value=False, )],
                                                                       style=dict(width='40%', display='table-cell')),
                                                              html.Div([dcc.RadioItems(id='TYPE_ras', options=[
                                                                  {'label': 'Butterworth', 'value': 'BTR'},
                                                                  {'label': 'Chebyshev', 'value': 'CBY'}, ],
                                                                                       value='BTR', labelStyle={
                                                                      'display': 'inline-block'}, inputStyle={
                                                                      "margin-left": "20px",
                                                                      "margin-right": "10px"}), ],
                                                                       style=dict(width='60%', display='table-cell')),
                                                          ], style=dict(width='100%', display='table')),
                                                          html.Div(html.P([html.Br()])),
                                                          html.H5('LFP Parameters:'),
                                                          html.Div([
                                                              html.Div(['Threshold (mV):'],
                                                                       style={'text-align': 'left', 'width': '60%',
                                                                              'display': 'table-cell'}),
                                                              html.Div([dcc.Input(id='prominence_ras',
                                                                                  placeholder='Threshold in (volts)',
                                                                                  type='number', value='0.07')],
                                                                       style={'text-align': 'right', 'width': '20%',
                                                                              'display': 'table-cell'}),
                                                          ], style=dict(width='100%', display='table')),
                                                          html.Div([
                                                              html.Div(['Time duration (Seconds):'],
                                                                       style={'text-align': 'left', 'width': '60%',
                                                                              'display': 'table-cell'}),
                                                              html.Div([dcc.Input(id='width_ras',
                                                                                  placeholder='Peak Width in (seconds)',
                                                                                  type='number', value='0.02')],
                                                                       style={'text-align': 'right', 'width': '20%',
                                                                              'display': 'table-cell'}),
                                                          ], style=dict(width='100%', display='table')),
                                                          html.Div(html.P([html.Br()])),
                                                          dcc.RadioItems(id='detect_mode_ras', options=[
                                                              {'label': 'Upward Peak Only', 'value': 'SS'},
                                                              {'label': 'Upward & Downward Peak', 'value': 'DS'}, ],
                                                                         value='DS',
                                                                         labelStyle={'display': 'inline-block'},
                                                                         inputStyle={"margin-left": "20px",
                                                                                     "margin-right": "10px"}),
                                                          html.Div(html.P([html.Br()])),
                                                          html.Div([
                                                              html.Div([html.Button(
                                                                  'Apply Settings and Generate Raster', id='button-3',
                                                                  n_clicks=0,style=tab_selected_style)],
                                                                  style=dict(width='100%', display='table-cell')),
                                                          ], style={'width':'100%', 'height':'100px', 'overflow-y':'scroll','display':'table'}),

                                                          html.H5('LFP Raster Analysis:'),
                                                          html.Div([
                                                              html.Div(['Group1: '],
                                                                       style=dict(width='20%', display='table-cell')),
                                                              html.Div(
                                                                  [dcc.Dropdown(id='group1', options=[], multi=True)],
                                                                  style=dict(width='70%', display='table-cell',  ))
                                                          ], style={'width':'100%', 'height':'100px', 'overflow-y':'scroll','display':'table'}),
                                                          html.Div([
                                                              html.Div(['Group2: '],
                                                                       style=dict(width='20%', display='table-cell')),
                                                              html.Div(
                                                                  [dcc.Dropdown(id='group2', options=[], multi=True)],
                                                                  style=dict(width='70%', display='table-cell', )),
                                                          ], style={'width':'100%', 'height':'100px', 'overflow-y':'scroll','display':'table'}),
                                                          html.Div([
                                                              html.Div(['Group3: '],
                                                                       style=dict(width='20%', display='table-cell')),
                                                              html.Div(
                                                                  [dcc.Dropdown(id='group3', options=[], multi=True)],
                                                                  style=dict(width='70%', display='table-cell', ))
                                                          ], style={'width':'100%', 'height':'100px', 'overflow-y':'scroll','display':'table'}),
                                                          html.Div(html.P([html.Br()])),
                                                          
                                                      ], ),
                                              ]),
                                              ]), ], style={'text-align': 'center','height':'700px', 'overflow-y':'scroll'}, className="three columns"),
                                ], className="row"),
                                html.Hr(
                                    style={'width': '5%', 'border-top': '2px dotted black', 'border-radius': '5px'}),

                                html.Div(children=[
                                    html.Div([
                                        dcc.Tabs(id='sz-analysis', value='lfp-plot', children=[
                                            dcc.Tab(label='LFP Activity View', value='lfp-plot', style=tab_style,
                                                    selected_style=tab_selected_style,
                                                    children=[

                                                        html.Div(html.P([html.Br()])),
                                                        html.Div([
                                                            html.Div(
                                                                [html.Div([
                                                                    html.Div(html.P([html.Br()])),
                                                                    html.Div([html.Div(
                                                                        [html.H6("Select a Time Range for Analysis: ")],
                                                                        style={'width': '20%',
                                                                               'vertical-align': 'middle',
                                                                               'display': 'table-cell'}),
                                                                              html.Div([dcc.RangeSlider(
                                                                                  id='my-range-slider-2', min=0, max=20,
                                                                                  value=[0, 10],
                                                                                  tooltip={'always_visible': True},
                                                                                  allowCross=False, dots=False)],
                                                                                       style={'width': '78%',
                                                                                              'vertical-align': 'middle',
                                                                                              'display': 'table-cell'}), ],
                                                                             style=dict(width='100%',
                                                                                        display='table'))]),
                                                                    html.Div(html.P([html.Br()])),
                                                                    html.Div([
                                                                        html.Div([html.H6(
                                                                            'Select a Channel to View Trace: ')],
                                                                                 style=dict(width='20%',
                                                                                            display='table-cell')),
                                                                        html.Div([dcc.Dropdown(id='ch_list', options=[],
                                                                                               placeholder='Select Active Channels from the Grid to View',
                                                                                               multi=True)],
                                                                                 style=dict(width='80%',
                                                                                            display='table-cell')), ],
                                                                        style=dict(width='100%', display='table'), ),
                                                                    html.Div([
                                                                        html.Div([dcc.Graph(id='true', figure=fig0)],className='six columns'),
                                                                        html.Div([dcc.Graph(id='fft', figure=fig0)],className='six columns'),],className='row'),
                                                                    ],
                                                                style={'text-align': 'center', 'width': '100%',
                                                                       'padding-left': '5%'},
                                                                className='ten columns'), ], className="row"),
                                                    ]),
                                            dcc.Tab(label='Seizure Activity View (Groups)', value='sz-plot',
                                                    style=tab_style,
                                                    selected_style=tab_selected_style,
                                                    children=[
                                                        dcc.Tabs([
                                                            dcc.Tab(label='Group-1', style=tab_style,
                                                                    selected_style=tab_selected_style,
                                                                    children=[
                                                                        html.H3([
                                                                                    "Seizure Detection and Analysis for Group-1 Channels"],
                                                                                style={'text-align': 'center',
                                                                                       'width': '100%',
                                                                                       'padding-left': '5%'}),
                                                                        html.Div(html.P([html.Br()])),
                                                                        html.Div([
                                                                            html.Div(['Convolution 1 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window1-g1',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='30')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 1 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff1-g1',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window2-g1',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='500')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff2-g1',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                        ], style=dict(width='100%', display='table')),
                                                                        html.Div([
                                                                            html.Div(children=[
                                                                                html.Div(html.P([html.Br()])),
                                                                                html.Div([
                                                                                    html.Div([
                                                                                        'Select a Channel to View Trace: '],
                                                                                        style=dict(width='60%',
                                                                                                   display='table-cell')),
                                                                                    html.Div([dcc.Dropdown(
                                                                                        id='ch_list-sz-g1', options=[],
                                                                                        multi=False)],
                                                                                             style=dict(width='40%',
                                                                                                        display='table-cell')), ],
                                                                                    style=dict(width='100%',
                                                                                               display='table'), ),
                                                                                html.Div([dcc.Graph(id='filt-g1',
                                                                                                    figure=fig0),
                                                                                html.Br(),html.H5("Summary Table of Measures"),
dash_table.DataTable(id='table-sz-gp1',columns=[
                                          {'name': 'sz #','id': 'sz_num','type': 'numeric'},
                                          {'name': 'start (s)','id': 'start','type': 'numeric'},
                                          {'name': 'end (s)','id': 'end', 'type': 'numeric'},
                                          {'name': 'LFP Count','id': 'count','type': 'numeric'},
                                          {'name': 'Mean Spike Frequency (Hz)','id': 'msf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Frequency (Hz)','id': 'm_isf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Interval (s)','id': 'mean_isi','type': 'numeric'},],
                                          data=[],style_as_list_view=True,
                     style_data={'border': '1px white','font_size': '14px','text_align': 'center'},
                     style_cell={'height': 'auto','minWidth': '50px','width': '100px','maxWidth': '100px','whiteSpace': 'normal'},
                     style_header={'backgroundColor': 'white','border': '1px white','font_size': '14px','fontWeight': 'bold','text-align': 'center'}),
                                                                                ],
                                                                                         style={'text-align': 'center',
                                                                                                'width': '100%'}),

                                                                            ], style={'text-align': 'center'},
                                                                                className="six columns"),

                                                                            html.Div(
                                                                                children=[html.Div(html.P([html.Br()])),
                                                                                          html.Div([
                                                                                                       "* Note: Multiply by Array Spacing in micrometer"],
                                                                                                   style={
                                                                                                       'text-align': 'center',
                                                                                                       'width': '100%'}),
                                                                                          html.Div([
                                                                                              dash_table.DataTable(
                                                                                                  id='table4',
                                                                                                  columns=[
                                                                                                      {
                                                                                                          'name': 'Time Interval [s]',
                                                                                                          'id': 'time-int',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Distance [no unit*]',
                                                                                                          'id': 'Distance',
                                                                                                          'type': 'numeric'},
                                                                                                      {
                                                                                                          'name': 'Max Duration [s]',
                                                                                                          'id': 'Max-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Mean Duration [s]',
                                                                                                          'id': 'Mean-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Seizure Rate [no unit*/s]',
                                                                                                          'id': 'sz-rate',
                                                                                                          'type': 'numeric'
                                                                                                      },

                                                                                                  ],
                                                                                                  data=[],
                                                                                                  style_as_list_view=True,
                                                                                                  style_data={
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '16px',
                                                                                                      'text_align': 'center'},
                                                                                                  style_cell={
                                                                                                      'height': 'auto',
                                                                                                      # all three widths are needed
                                                                                                      'minWidth': '50px',
                                                                                                      'width': '100px',
                                                                                                      'maxWidth': '100px',
                                                                                                      'whiteSpace': 'normal'
                                                                                                  },
                                                                                                  style_header={
                                                                                                      'backgroundColor': 'white',
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '18px',
                                                                                                      'fontWeight': 'bold',
                                                                                                      'text-align': 'center'})],
                                                                                              style=dict(width='100%',
                                                                                                         display='table')),
                                                                                          html.Div(html.P([html.Br()])),
                                                                                          dcc.Graph(id='path-g1',
                                                                                                    figure=fig0)],
                                                                                style={'text-align': 'center',
                                                                                       'padding-left': '5%'},
                                                                                className="six columns"),
                                                                        ]), ], className="row"),

                                                            dcc.Tab(label='Group-2', style=tab_style,
                                                                    selected_style=tab_selected_style,
                                                                    children=[
                                                                        html.H3([
                                                                                    "Seizure Detection and Analysis for Group-2 Channels"],
                                                                                style={'text-align': 'center',
                                                                                       'width': '100%',
                                                                                       'padding-left': '5%'}),
                                                                        html.Div(html.P([html.Br()])),
                                                                        html.Div([
                                                                            html.Div(['Convolution 1 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window1-g2',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='30')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 1 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff1-g2',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window2-g2',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='500')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff2-g2',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                        ], style=dict(width='100%', display='table')),
                                                                        html.Div([
                                                                            html.Div(children=[
                                                                                html.Div(html.P([html.Br()])),
                                                                                html.Div([
                                                                                    html.Div([
                                                                                        'Select a Channel to View Trace: '],
                                                                                        style=dict(width='60%',
                                                                                                   display='table-cell')),
                                                                                    html.Div([dcc.Dropdown(
                                                                                        id='ch_list-sz-g2', options=[],
                                                                                        multi=False)],
                                                                                             style=dict(width='40%',
                                                                                                        display='table-cell')), ],
                                                                                    style=dict(width='100%',
                                                                                               display='table'), ),
                                                                                html.Div([dcc.Graph(id='filt-g2',
                                                                                                    figure=fig0),
                                                                                 html.Br(),html.H5("Summary Table of Measures"),
dash_table.DataTable(id='table-sz-gp2',columns=[
                                          {'name': 'sz #','id': 'sz_num','type': 'numeric'},
                                          {'name': 'start (s)','id': 'start','type': 'numeric'},
                                          {'name': 'end (s)','id': 'end', 'type': 'numeric'},
                                          {'name': 'LFP Count','id': 'count','type': 'numeric'},
                                          {'name': 'Mean Spike Frequency (Hz)','id': 'msf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Frequency (Hz)','id': 'm_isf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Interval (s)','id': 'mean_isi','type': 'numeric'},],
                                          data=[],style_as_list_view=True,
                     style_data={'border': '1px white','font_size': '14px','text_align': 'center'},
                     style_cell={'height': 'auto','minWidth': '50px','width': '100px','maxWidth': '100px','whiteSpace': 'normal'},
                     style_header={'backgroundColor': 'white','border': '1px white','font_size': '14px','fontWeight': 'bold','text-align': 'center'}),],
                                                                                         style={'text-align': 'center',
                                                                                                'width': '100%'}),

                                                                            ], style={'text-align': 'center'},
                                                                                className="six columns"),

                                                                            html.Div(
                                                                                children=[html.Div(html.P([html.Br()])),
                                                                                          html.Div([
                                                                                                       "* Note: Multiply by Array Spacing in micrometer"],
                                                                                                   style={
                                                                                                       'text-align': 'center',
                                                                                                       'width': '100%'}),
                                                                                          html.Div([
                                                                                              dash_table.DataTable(
                                                                                                  id='table5',
                                                                                                  columns=[
                                                                                                      {
                                                                                                          'name': 'Time Interval [s]',
                                                                                                          'id': 'time-int',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Distance [no unit*]',
                                                                                                          'id': 'Distance',
                                                                                                          'type': 'numeric'},
                                                                                                      {
                                                                                                          'name': 'Max Duration [s]',
                                                                                                          'id': 'Max-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Mean Duration [s]',
                                                                                                          'id': 'Mean-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Seizure Rate [no unit*/s]',
                                                                                                          'id': 'sz-rate',
                                                                                                          'type': 'numeric'
                                                                                                      },

                                                                                                  ],
                                                                                                  data=[],
                                                                                                  style_as_list_view=True,
                                                                                                  style_data={
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '16px',
                                                                                                      'text_align': 'center'},
                                                                                                  style_cell={
                                                                                                      'height': 'auto',
                                                                                                      'minWidth': '50px',
                                                                                                      'width': '100px',
                                                                                                      'maxWidth': '100px',
                                                                                                      'whiteSpace': 'normal'
                                                                                                  },
                                                                                                  style_header={
                                                                                                      'backgroundColor': 'white',
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '18px',
                                                                                                      'fontWeight': 'bold',
                                                                                                      'text-align': 'center'})],
                                                                                              style=dict(width='100%',
                                                                                                         display='table')),
                                                                                          html.Div(html.P([html.Br()])),
                                                                                          dcc.Graph(id='path-g2',
                                                                                                    figure=fig0)],
                                                                                style={'text-align': 'center',
                                                                                       'padding-left': '5%'},
                                                                                className="six columns"),
                                                                        ]), ]),
                                                            dcc.Tab(label='Group-3', style=tab_style,
                                                                    selected_style=tab_selected_style,
                                                                    children=[
                                                                        html.H3([
                                                                                    "Seizure Detection and Analysis for Group-3 Channels"],
                                                                                style={'text-align': 'center',
                                                                                       'width': '100%',
                                                                                       'padding-left': '5%'}),
                                                                        html.Div(html.P([html.Br()])),
                                                                        html.Div([
                                                                            html.Div(['Convolution 1 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window1-g3',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='30')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 1 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff1-g3',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Window):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='window2-g3',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='500')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div(['Convolution 2 (Cut-off):'],
                                                                                     style={'text-align': 'center',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                            html.Div([dcc.Input(id='cutoff2-g3',
                                                                                                placeholder='Peak Width in (seconds)',
                                                                                                type='number',
                                                                                                debounce=True,
                                                                                                value='0.04')],
                                                                                     style={'text-align': 'left',
                                                                                            'width': '15%',
                                                                                            'display': 'table-cell'}),
                                                                        ], style=dict(width='100%', display='table')),
                                                                        html.Div([
                                                                            html.Div(children=[
                                                                                html.Div(html.P([html.Br()])),
                                                                                html.Div([
                                                                                    html.Div([
                                                                                        'Select a Channel to View Trace: '],
                                                                                        style=dict(width='60%',
                                                                                                   display='table-cell')),
                                                                                    html.Div([dcc.Dropdown(
                                                                                        id='ch_list-sz-g3', options=[],
                                                                                        multi=False)],
                                                                                             style=dict(width='40%',
                                                                                                        display='table-cell')), ],
                                                                                    style=dict(width='100%',
                                                                                               display='table'), ),
                                                                                html.Div([dcc.Graph(id='filt-g3',
                                                                                                    figure=fig0),
                                                                                 html.Br(),html.H5("Summary Table of Measures"),
dash_table.DataTable(id='table-sz-gp3',columns=[
                                          {'name': 'sz #','id': 'sz_num','type': 'numeric'},
                                          {'name': 'start (s)','id': 'start','type': 'numeric'},
                                          {'name': 'end (s)','id': 'end', 'type': 'numeric'},
                                          {'name': 'LFP Count','id': 'count','type': 'numeric'},
                                          {'name': 'Mean Spike Frequency (Hz)','id': 'msf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Frequency (Hz)','id': 'm_isf','type': 'numeric'},
                                          {'name': 'Mean Inter Spike Interval (s)','id': 'mean_isi','type': 'numeric'},],
                                          data=[],style_as_list_view=True,
                     style_data={'border': '1px white','font_size': '14px','text_align': 'center'},
                     style_cell={'height': 'auto','minWidth': '50px','width': '100px','maxWidth': '100px','whiteSpace': 'normal'},
                     style_header={'backgroundColor': 'white','border': '1px white','font_size': '14px','fontWeight': 'bold','text-align': 'center'}),],
                                                                                         style={'text-align': 'center',
                                                                                                'width': '100%'}),

                                                                            ], style={'text-align': 'center'},
                                                                                className="six columns"),

                                                                            html.Div(
                                                                                children=[html.Div(html.P([html.Br()])),
                                                                                          html.Div([
                                                                                                       "* Note: Multiply by Array Spacing in micrometer"],
                                                                                                   style={
                                                                                                       'text-align': 'center',
                                                                                                       'width': '100%'}),
                                                                                          html.Div([
                                                                                              dash_table.DataTable(
                                                                                                  id='table6',
                                                                                                  columns=[
                                                                                                      {
                                                                                                          'name': 'Time Interval [s]',
                                                                                                          'id': 'time-int',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Distance [no unit*]',
                                                                                                          'id': 'Distance',
                                                                                                          'type': 'numeric'},
                                                                                                      {
                                                                                                          'name': 'Max Duration [s]',
                                                                                                          'id': 'Max-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Mean Duration [s]',
                                                                                                          'id': 'Mean-Duration',
                                                                                                          'type': 'numeric'
                                                                                                      },
                                                                                                      {
                                                                                                          'name': 'Seizure Rate [no unit*/s]',
                                                                                                          'id': 'sz-rate',
                                                                                                          'type': 'numeric'
                                                                                                      },

                                                                                                  ],
                                                                                                  data=[],
                                                                                                  style_as_list_view=True,
                                                                                                  style_data={
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '16px',
                                                                                                      'text_align': 'center'},
                                                                                                  style_cell={
                                                                                                      'height': 'auto',
                                                                                                      # all three widths are needed
                                                                                                      'minWidth': '50px',
                                                                                                      'width': '100px',
                                                                                                      'maxWidth': '100px',
                                                                                                      'whiteSpace': 'normal'
                                                                                                  },
                                                                                                  style_header={
                                                                                                      'backgroundColor': 'white',
                                                                                                      'border': '2px grey',
                                                                                                      'font_size': '18px',
                                                                                                      'fontWeight': 'bold',
                                                                                                      'text-align': 'center'})],
                                                                                              style=dict(width='100%',
                                                                                                         display='table')),
                                                                                          html.Div(html.P([html.Br()])),
                                                                                          dcc.Graph(id='path-g3',
                                                                                                    figure=fig0)],
                                                                                style={'text-align': 'center',
                                                                                       'padding-left': '5%'},
                                                                                className="six columns"),
                                                                        ]), ]),
                                                        ]),

                                                    ], ),
                                        ], ),
                                    ], className='twelve columns'), ], className="row"),
                                html.Div(html.P([html.Br(), html.Br()])),
                                ])


@app.callback(
    [Output('file_name_text', 'children'), Output('g2', 'figure'), Output('table', 'data'),
     Output('my-range-slider', 'min'), Output('my-range-slider', 'max'), Output('my-range-slider', 'value'),
     Output('my-range-slider', 'marks'),Output('my-range-slider-2', 'min'), Output('my-range-slider-2', 'max'),
     Output('my-range-slider-2', 'value'), Output('my-range-slider-2', 'marks'), Output('g2_g1', 'figure'), 
     Output('g2_g2', 'figure'),Output('g2_g3', 'figure')],
    [Input('submit-file', 'n_clicks'), Input('upload-image', 'contents')],
    [State('File-Name', 'value'), State('upload-image', 'filename'), ])
def update_grid(n_clicks, img, value, img_file):
    bool = check_filename(value)

    if value and bool == True:
        filepath = value
        h5 = h5py.File(filepath, 'r')
        parameters = parameter(h5)
        chsList = parameters['recElectrodeList']
        Frames = parameters['nRecFrames']
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chsList))
        x_label = np.linspace(1, 64, 64)
        y_label = np.linspace(1, 64, 64)
        xx, yy = np.meshgrid(x_label, y_label, sparse=False, indexing='xy')
        row_list, column_list, row_list_noise, column_list_noise = get_row_column_list(data, chsList, parameters)
        range_min = 0
        range_max = (parameters['nRecFrames'] / parameters['samplingRate']) - 10
        range_steps = 12
        marks = range_slider_marks(range_min, range_max, range_steps)

        file_name = filepath.split('\\')[-1]
        file_path = '\\'.join(filepath.split('\\')[0:-1])

        info_dict = {'Filename': str(filepath), 'Channels': str(len(chsList)), 'Frames': str(parameters['nRecFrames']),
                     'Recording-Length': str(round(parameters['nRecFrames'] / parameters['samplingRate'])),
                     'Sampling': str(parameters['samplingRate'])}
        table_dict = [{'File-Path': str(file_path), 'File-Name': str(file_name), 'Total-Active-Channels': len(chsList),
                       'Frames': parameters['nRecFrames'],
                       'Recording-Length': round(parameters['nRecFrames'] / parameters['samplingRate']),
                       'Sampling-Rate': parameters['samplingRate']}, ]

        image_filename = img_file
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=column_list, y=row_list, marker={'color': 'green', 'showscale': False}, mode='markers',
                    name='Active Channels'))
        fig2.add_trace(go.Scatter(x=column_list_noise, y=row_list_noise, marker={'color': 'black', 'showscale': False},
                    mode='markers',name='Noise Channels'))
        fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
        fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65],
                    autorange="reversed")
        fig2.update_layout(images=[dict(source=img,xref="paper",yref="paper",x=0,y=1,sizex=1,sizey=1,sizing="stretch",opacity=0.5,layer="above")],)
        fig2.update_layout(template="plotly_white", showlegend=True, clickmode='event+select', width=600, height=600,legend=dict(orientation="h"))

        output = file_path + '\\results-' + file_name.split('.')[0]

        if not os.path.exists(output):
            os.makedirs(output)

        csv_file_name = output + '\\' + 'group_summary_log.csv'
        
        if not os.path.exists(csv_file_name):
            df_groups = pd.DataFrame(columns=['Time-Stamp', 'Time-Window', 'Group', 'LFP-Count', 'Tot-Channel', 'Act-Channel',
                         'LFP-Count-perCH', 'LFP-Count-per-Time', 'time-first-event', 'SZ-Channels', 'SZ-start',
                         'SZ-duration', 'SZ-distance', 'SZ-rate'])
            df_groups.to_csv(csv_file_name, header=True, index=False)

    elif value and bool == False:
        path = 'Enter a Valid BrainWave4 .brw File (Uncompressed Exported - RAW)'
        info_dict = {'Message': path}
        table_dict = [{'File-Path': '', 'File-Name': '', 'Total-Active-Channels': 0, 'Frames': 0, 'Recording-Length': 0,
                       'Sampling-Rate': 0}, ]
        fig2 = create_sensor_grid()
        range_min = 0
        range_max = 20
        range_steps = 20
        marks = range_slider_marks(range_min, range_max, range_steps)

    else:
        path = "Enter Complete File-Path and Filename to Begin [Example: C:/Users/amahadevan/Documents/slice1_resample.brw]"
        info_dict = {'Message': path}
        table_dict = [{'File-Path': '', 'File-Name': '', 'Total-Active-Channels': 0, 'Frames': 0, 'Recording-Length': 0,
                       'Sampling-Rate': 0}, ]
        fig2 = create_sensor_grid()
        range_min = 0
        range_max = 20
        range_steps = 20
        marks = range_slider_marks(range_min, range_max, range_steps)

    return html.H6(json.dumps(info_dict)), fig2, table_dict, range_min, range_max, [(range_min + int(range_max / 3)),
                                                                                    range_max / 2], marks, range_min, range_max, [
               (range_min + int(range_max / 3)), range_max / 2], marks, fig2, fig2, fig2


@app.callback(
    [Output('g3', 'figure'), Output('g9', 'figure'), Output('lfp', 'value'), Output('amplitude', 'value'),Output('duration', 'value')],
    [Input('button-2', 'n_clicks'), Input('file_name_text', 'children')],
    [State('my-range-slider', 'value'), State('prominence', 'value'), State('width', 'value'),State('detect_mode', 'value')])
def update_peak_raster(n_clicks1, value, range_value, prom, wid, detect_mode):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    if 'Filename' in path0.keys() and check_filename(path0['Filename']) == True:
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = parameters['recElectrodeList']
        Frames = int(path0['Frames'])
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chsList))
        frame0 = int(range_value[0] * parameters['samplingRate'])
        frame1 = int(range_value[1] * parameters['samplingRate'])
        frame_value = [frame0, frame1]
        data = data[frame0:frame1, :]

        column_list = get_channel_list(chsList)
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        interval = float(wid) * float(parameters['samplingRate'])
        interval = int(interval)
        fig3, fig9, mean_LFP, mean_amplitude, mean_width = peak_raster_grid(data, column_list, parameters,float(prom), 
                                                                            interval,detect_mode, frame_value)
        h5.close()

        output = '\\'.join(path0['Filename'].split('\\')[0:-1]) + '\\results-' + \
                 path0['Filename'].split('\\')[-1].split('.')[0]
        if not os.path.exists(output):
            os.makedirs(output)


        return fig3, fig9, round(mean_LFP, 4), round(mean_amplitude, 4), round(mean_width, 4)

    else:
        return fig, fig, 0, 0, 0


@app.callback(
    [Output('g7', 'figure'), Output('g9_ch', 'figure'), Output('table2', 'data'), Output('g1_GRID','figure'),Output('g2_GRID','figure'),Output('g3_GRID','figure')],
    [Input('button-3', 'n_clicks'), Input('file_name_text', 'children')],
    [State('my-range-slider', 'value'), State('my-toggle-switch_ras', 'value'), State('lower_ras', 'value'),
     State('upper_ras', 'value'),
     State('TYPE_ras', 'value'), State('prominence_ras', 'value'), State('width_ras', 'value'),
     State('detect_mode_ras', 'value'),
     State('group1', 'value'), State('group2', 'value'), State('group3', 'value')])
def update_channel_raster(n_clicks2, value, range_value, toggle, lower, upper, type, prom, wid, detect_mode, group_1,group_2, group_3):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    ctx = dash.callback_context
    trigger = ctx.triggered
    button = trigger[0]['prop_id']

    if 'Filename' in path0.keys() and check_filename(path0['Filename']) == True and button == 'button-3.n_clicks':
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        frame0 = int(range_value[0] * parameters['samplingRate'])
        frame1 = int(range_value[1] * parameters['samplingRate'])

        chsList_row_col = parameters['recElectrodeList']
        row_list = []
        column_list = []
        for item in chsList_row_col:
            row_list.append(item['Row'])
            column_list.append(item['Col'])

        Frames = int(path0['Frames'])
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chsList))[frame0:frame1, :]
        interval = float(wid) * float(parameters['samplingRate'])
        interval = int(interval)
        group_dict = {0: group_1, 1: group_2, 2: group_3}
        x = np.arange(0, parameters['nRecFrames'], 1) / parameters['samplingRate']
        x = x[frame0:frame1]
        fig7 = go.Figure()
        color = {0: 'Crimson', 1: 'MediumPurple', 2: 'Green'}
        channels_total = 0

        for i in range(3):
            if group_dict[i]:
                channels_total += len(group_dict[i])

        lfp = np.zeros(parameters['nRecFrames'])
        pp_prominence = np.zeros(parameters['nRecFrames'])
        pp_width = np.zeros(parameters['nRecFrames'])

        channel_list = []
        rows = []
        columns = []
        group_num = []
        lfp_count_ch = []
        avg_amplitude_ch = []
        avg_duration_ch = []
        time_to_event = []

        df_channel = pd.DataFrame(columns=['ch_num', 'row_num', 'column_num', 'group_number', 'lfp_count', 
                                            'avg_amplitude', 'avg_duration','time_to_event'])
        df_groups = pd.DataFrame(columns=['Group', 'LFP-Count', 'Tot-Channel', 'Act-Channel', 'LFP-Count20',
                                            'LFP-Count-CH', 'LFP-Count-Time','Mean-Amplitude', 'Mean-Duration'])
        df_groups['Group'] = [1, 2, 3]

        for i in range(3):
            group = group_dict[i]
            if group and len(group) > 0:
                count = 0
                for ch in group:
                    gp = int(i)
                    group_num.append(gp)
                    lfp_sub = np.zeros(parameters['nRecFrames'])
                    pp_prominence_sub = np.zeros(parameters['nRecFrames'])
                    pp_width_sub = np.zeros(parameters['nRecFrames'])
                    row, column = get_row_col_num(int(ch))
                    ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
                    ch_y = data[:, ch_id]
                    CH_Y = convert_to_uV(ch_y, parameters) / 1000000
                    CH_Y = (CH_Y - np.mean(CH_Y))
                    if toggle == True:
                        sig = frequency_filter(CH_Y, parameters['samplingRate'], type, int(lower), int(upper), order=6)
                    else:
                        sig = CH_Y

                    if detect_mode == 'DS':
                        sig = abs(sig)
                    else:
                        sig = sig
                    peaks, properties = find_peaks(sig, prominence=float(prom), width=interval)
                    spikes = np.arange(0, parameters['nRecFrames'], 1)[peaks]
                    channel_list.append(ch)
                    rows.append(row)
                    columns.append(column)
                    lfp_sub[peaks] = 1
                    pp_prominence_sub[peaks] = abs(sig)[peaks]
                    pp_mean = np.mean(abs(sig)[peaks])
                    pw_mean = np.mean(properties['widths'])
                    index_pp = np.where(pp_prominence_sub >= (pp_mean))
                    pp_width_sub[peaks] = properties['widths'] / parameters['samplingRate']
                    index_pw = np.where(pp_width_sub >= pw_mean / parameters['samplingRate'])
                    if len(index_pp[0]) > 0 and len(index_pw[0]) > 0:
                        time_to_event.append(x[index_pp[0][0]])
                    else:
                        time_to_event.append(x[-1])
                    lfp += lfp_sub
                    pp_prominence += pp_prominence_sub
                    pp_width += pp_width_sub
                    lfp_count_ch.append(np.sum(lfp_sub))
                    avg_amplitude_ch.append(np.mean(pp_prominence_sub[peaks]))
                    avg_duration_ch.append(np.mean(pp_width_sub[peaks]))

                    label1 = '(' + str(column) + ', ' + str(row) + ')'
                    label = "CH-" + label1
                    spikes_label = [label for _ in spikes]
                    fig7.add_trace(go.Scatter(x=x[peaks], y=spikes_label, mode='markers', marker_size=2, marker_color=color[i],
                                                name=label, legendgroup="Group " + str(i + 1)))
                    fig7.update_layout(template="plotly_white", showlegend=True, width=600, height=600)
                    count += 1
            else:
                pass

        output = '\\'.join(path0['Filename'].split('\\')[0:-1]) + '\\results-' + \
                 path0['Filename'].split('\\')[-1].split('.')[0]
        if not os.path.exists(output):
            os.makedirs(output)
        csv_file_name = output + '\\' + 'summary_all.csv'

        df_channel['ch_num'] = channel_list
        df_channel['row_num'] = rows
        df_channel['column_num'] = columns
        df_channel['group_number'] = group_num
        df_channel['lfp_count'] = lfp_count_ch
        df_channel['avg_amplitude'] = avg_amplitude_ch
        df_channel['avg_duration'] = avg_duration_ch
        df_channel['time_to_event'] = time_to_event
        csv_file_name = output + '' + 'summary_all.csv'
        df_channel.to_csv(csv_file_name, header=True, index=False)

        tot_channel_0 = len(df_channel[df_channel['group_number'] == 0])
        tot_channel_1 = len(df_channel[df_channel['group_number'] == 1])
        tot_channel_2 = len(df_channel[df_channel['group_number'] == 2])

        g1_ = df_channel[df_channel['group_number'] == 0]
        g2_ = df_channel[df_channel['group_number'] == 1]
        g3_ = df_channel[df_channel['group_number'] == 2]

        g1_column = list(g1_['column_num'])
        g1_row = list(g1_['row_num'])
        g2_column = list(g2_['column_num'])
        g2_row = list(g2_['row_num'])
        g3_column = list(g3_['column_num'])
        g3_row = list(g3_['row_num'])

        df_channel = df_channel[df_channel['lfp_count'] > 0]
        df_channel_0 = df_channel[df_channel['group_number'] == 0]
        df_channel_1 = df_channel[df_channel['group_number'] == 1]
        df_channel_2 = df_channel[df_channel['group_number'] == 2]

        df_channel_0 = df_channel_0.sort_values(by=['lfp_count'], ascending = False)
        df_channel_1 = df_channel_1.sort_values(by=['lfp_count'], ascending = False)
        df_channel_2 = df_channel_2.sort_values(by=['lfp_count'], ascending = False)

        g1_act_column = list(df_channel_0['column_num'])
        g1_act_row = list(df_channel_0['row_num'])
        g1_GRID = get_grid(column_list,row_list, g1_act_column,g1_act_row, g1_column,g1_row)
        g2_act_column = list(df_channel_1['column_num'])
        g2_act_row = list(df_channel_1['row_num'])
        g2_GRID = get_grid(column_list,row_list, g2_act_column,g2_act_row, g2_column,g2_row)
        g3_act_column = list(df_channel_2['column_num'])
        g3_act_row = list(df_channel_2['row_num'])
        g3_GRID = get_grid(column_list,row_list, g3_act_column,g3_act_row,g2_column,g2_row)

        act_channel_0 = len(df_channel_0)
        act_channel_1 = len(df_channel_1)
        act_channel_2 = len(df_channel_2)

        df_groups['LFP-Count'].loc[0] = round(np.sum(np.array(df_channel_0['lfp_count']))/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count'].loc[1] = round(np.sum(np.array(df_channel_1['lfp_count']))/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count'].loc[2] = round(np.sum(np.array(df_channel_2['lfp_count']))/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count20'].loc[0] = round(np.sum(np.array(df_channel_0['lfp_count'])[0:20])/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count20'].loc[1] = round(np.sum(np.array(df_channel_1['lfp_count'])[0:20])/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count20'].loc[2] = round(np.sum(np.array(df_channel_2['lfp_count'])[0:20])/(range_value[1] - range_value[0]),2)
        df_groups['LFP-Count-CH'].loc[0] = round(df_groups['LFP-Count'].loc[0] / act_channel_0, 0)
        df_groups['LFP-Count-CH'].loc[1] = round(df_groups['LFP-Count'].loc[1] / act_channel_1, 0)
        df_groups['LFP-Count-CH'].loc[2] = round(df_groups['LFP-Count'].loc[2] / act_channel_2, 0)
        df_groups['LFP-Count-Time'].loc[0] = round(df_groups['LFP-Count'].loc[0] / (range_value[1] - range_value[0]) / act_channel_0, 2)
        df_groups['LFP-Count-Time'].loc[1] = round(df_groups['LFP-Count'].loc[1] / (range_value[1] - range_value[0]) / act_channel_1, 2)
        df_groups['LFP-Count-Time'].loc[2] = round(df_groups['LFP-Count'].loc[2] / (range_value[1] - range_value[0]) / act_channel_2, 2)
        df_groups['Tot-Channel'].loc[0] = tot_channel_0
        df_groups['Tot-Channel'].loc[1] = tot_channel_1
        df_groups['Tot-Channel'].loc[2] = tot_channel_2
        df_groups['Act-Channel'].loc[0] = act_channel_0
        df_groups['Act-Channel'].loc[1] = act_channel_1
        df_groups['Act-Channel'].loc[2] = act_channel_2
        df_groups['Mean-Amplitude'].loc[0] = round(np.mean(np.array(df_channel_0['avg_amplitude'])[0:20]), 3)
        df_groups['Mean-Amplitude'].loc[1] = round(np.mean(np.array(df_channel_1['avg_amplitude'])[0:20]), 3)
        df_groups['Mean-Amplitude'].loc[2] = round(np.mean(np.array(df_channel_2['avg_amplitude'])[0:20]), 3)
        df_groups['Mean-Duration'].loc[0] = round(np.mean(np.array(df_channel_0['avg_duration'])[0:20]), 3)
        df_groups['Mean-Duration'].loc[1] = round(np.mean(np.array(df_channel_1['avg_duration'])[0:20]), 3)
        df_groups['Mean-Duration'].loc[2] = round(np.mean(np.array(df_channel_2['avg_duration'])[0:20]), 3)
        summary_dict = df_groups.to_dict("rows")

        fig10 = go.Figure()
        axis_template = dict(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,mirror=True)
        fig10 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,subplot_titles=('Count', 'Amplitude', 'Duration'))
        fig10.update_layout(template="plotly_white", xaxis=axis_template, yaxis=axis_template, showlegend=False,width=600, height=500, title_text='LFP Events')
        time = np.arange(0, parameters['nRecFrames'], 1) / parameters['samplingRate']
        time = time[frame0:frame1]

        fig10.add_trace(go.Scatter(x=time, y=lfp), row=1, col=1)
        fig10.add_trace(go.Scatter(x=time, y=pp_prominence), row=2, col=1)
        fig10.add_trace(go.Scatter(x=time, y=pp_width), row=3, col=1)

        mean_LFP = np.sum(lfp) * parameters['samplingRate'] / parameters['nRecFrames']
        mean_amplitude = np.sum(pp_prominence) * parameters['samplingRate'] / parameters['nRecFrames']
        mean_width = np.sum(pp_width) * parameters['samplingRate'] / parameters['nRecFrames']


        fig10.update_xaxes(showgrid=False, linecolor='black', showticklabels=False, linewidth=2, showline=True,mirror=True, row=1, col=1)
        fig10.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,mirror=True, row=1, col=1)
        fig10.update_xaxes(showgrid=False, linecolor='black', showticklabels=False, linewidth=2, showline=True,mirror=True, row=2, col=1)
        fig10.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,mirror=True, row=2, col=1)
        fig10.update_xaxes(title_text='Time, Seconds', showgrid=False, linecolor='black', showticklabels=True,linewidth=2, showline=True,mirror=True, row=3, col=1)
        fig10.update_yaxes(showgrid=False, linecolor='black', showticklabels=True, linewidth=2, showline=True,mirror=True, row=3, col=1)

        df_save = df_channel_0
        df_save = df_save.append(df_channel_1)
        df_save = df_save.append(df_channel_2)
        file_path_output = output + '\\' + "data_channels.mat"
        csv_file_name = output + '\\' + 'summary.csv'
        df_save.to_csv(csv_file_name, header=True, index=False)

        fig7.update_xaxes(showline=True, linewidth=1, showgrid=False, linecolor='black', mirror=True,title_text='Time, Seconds')
        fig7.update_yaxes(showline=True, linewidth=1, showgrid=False, linecolor='black', mirror=True)
        h5.close()

        return fig7, fig10, summary_dict, g1_GRID, g2_GRID, g3_GRID


    else:
        return fig4, fig, [], fig, fig,fig


@app.callback(
    Output('ch_list', 'options'),
    [Input('g2', 'selectedData')])
def display_selected_ch(selectedData):
    points = selectedData
    if points is None:
        default = []
        return default
    else:
        ctx = dash.callback_context
        channels = []
        options = []
        for item in points["points"]:
            option_dict = {}
            point = (item['y'], item['x'])
            option_dict['label'] = str(item['y']) + ', ' + str(item['x'])
            option_dict['value'] = get_ch_number(item['y'], item['x'])
            options.append(option_dict)
            channels.append(point)
        return options


@app.callback(
    [Output('group1', 'options'), Output('group2', 'options'), Output('group3', 'options'), Output('group1', 'value'),
     Output('group2', 'value'), Output('group3', 'value'), Output('ch_list-sz-g1', 'options'),
     Output('ch_list-sz-g2', 'options'), Output('ch_list-sz-g3', 'options')],
    [Input('g2_g1', 'selectedData'), Input('g2_g2', 'selectedData'), Input('g2_g3', 'selectedData')])
def display_chs_raster_group(selectedData1, selectedData2, selectedData3):
    points1 = selectedData1
    points2 = selectedData2
    points3 = selectedData3

    def options_ch(points):
        channels = []
        options = []
        option_values = []
        for item in points["points"]:
            option_dict = {}
            point = (item['y'], item['x'])
            option_dict['label'] = str(item['y']) + ', ' + str(item['x'])
            option_dict['value'] = get_ch_number(item['y'], item['x'])
            options.append(option_dict)
            channels.append(point)
            option_values.append(get_ch_number(item['y'], item['x']))

        return option_values, options

    if points1 is None:
        options1 = []
        option_values1 = []
    else:
        option_values1, options1 = options_ch(points1)

    if points2 is None:
        options2 = []
        option_values2 = []
    else:
        option_values2, options2 = options_ch(points2)

    if points3 is None:
        options3 = []
        option_values3 = []
    else:
        option_values3, options3 = options_ch(points3)

    return options1, options2, options3, option_values1, option_values2, option_values3, options1, options2, options3


@app.callback(
    Output('true', 'figure'),
    [Input('button-2', 'n_clicks'), Input('ch_list', 'value'), Input('file_name_text', 'children'),
     Input('my-range-slider-2', 'value')],
    [State('my-toggle-switch', 'value'), State('lower', 'value'), State('upper', 'value'),
     State('TYPE', 'value'), State('prominence', 'value'), State('width', 'value'), State('detect_mode', 'value')])
def update_figure(n_clicks, ch_value, value, range_value, toggle, lower, upper, type, prom, wid, detect_mode):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    ch_id = ch_value
    if ch_id is None or 'Filename' not in path0.keys() or check_filename(path0['Filename']) == False:
        return fig
    else:
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        tot_chs = parameters['numRecElectrodes']
        Frames = int(path0['Frames'])
        ch_x = np.linspace(0, Frames, Frames) / parameters['samplingRate']
        frame0 = int(range_value[0] * parameters['samplingRate'])
        frame1 = int(range_value[1] * parameters['samplingRate'])
        channels = []
        for item in ch_value:
            channels.append(item)
 
        ch_x = ch_x[frame0:frame1]

        if len(channels)<2:
            width_plot = 500
        else:
            width_plot = 300

        if len(channels) > 0:
            fig2 = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                 x_title="Time, Seconds", y_title="Voltage, (mV)")
            plot = 0
            for item in channels:
                row, column = get_row_col_num(int(item))
                ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
                data = np.array(h5['/3BData/Raw']).reshape(Frames, tot_chs)
                ch_y = data[frame0:frame1, ch_id]
                CH_Y = convert_to_uV(ch_y, parameters) / 1000000
                CH_Y = (CH_Y - np.mean(CH_Y))
                label = ('(' + str(row) + ',' + str(column) + ')')
                interval = float(wid) * float(parameters['samplingRate'])
                interval = int(interval)
                if detect_mode == 'DS':
                    sig = abs(CH_Y)
                else:
                    sig = CH_Y
                peaks, properties = find_peaks(sig, prominence=float(prom), width=interval)
                pp = np.array(properties["prominences"])
                pp = np.zeros(len(properties["prominences"]))+max(CH_Y)*1.2
                time_axis = ch_x[peaks]
                if toggle == False:
                    fig2.add_trace(go.Scatter(x=ch_x, y=CH_Y, mode='lines', name=label), row=plot + 1, col=1)
                    fig2.add_trace(go.Scattergl(x=time_axis, y=pp, mode='markers', name=label, marker_size=20, marker_symbol='line-ns-open'), row=plot + 1, col=1)
                    fig2.update_layout(height=len(channels) * width_plot, width=1000, title_text="True")
                    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True)
                else:
                    CH_Y_f = frequency_filter(CH_Y, parameters['samplingRate'], type, int(lower), int(upper), order=6)
                    fig2.update_layout(height=len(channels) * width_plot, width=1000, title_text="Filtered")
                    fig2.add_trace(go.Scatter(x=ch_x, y=CH_Y, mode='lines', name=label), row=plot + 1, col=1)
                    fig2.add_trace(go.Scattergl(x=time_axis, y=pp, mode='markers', name=label, marker_size=20, marker_symbol='line-ns-open'), row=plot + 1, col=1)
                    fig2.add_trace(go.Scatter(x=ch_x, y=CH_Y_f, mode='lines', name=label + 'filter'), row=plot + 1, col=1)
                    fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, showticklabels=True)
                    
                    if detect_mode == 'DS':
                        sig_f = abs(CH_Y_f)
                    else:
                        sig_f = CH_Y_f
                    peaks_ff, properties_ff = find_peaks(sig_f, prominence=float(prom), width=interval)
                    pp_f = np.array(properties_ff["prominences"])
                    pp_f = np.zeros(len(properties_ff["prominences"]))+max(sig_f)*1.2
                    time_axis_f = ch_x[peaks_ff]
                    fig2.add_trace(go.Scattergl(x=time_axis_f, y=pp_f, mode='markers', name=label + 'filter', marker_size=20, marker_symbol='line-ns-open'), row=plot + 1, col=1)
                fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
                fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
                fig2.update_layout(template="plotly_white", showlegend=True, legend=dict(orientation="h"))
                plot += 1

 
            h5.close()
            return fig2

        else:
            return fig


@app.callback(
    Output('fft', 'figure'),
    [Input('button-2','n_clicks'),Input('ch_list', 'value'),Input('true', 'relayoutData'),Input('file_name_text', 'children')],[State('my-toggle-switch', 'value'),State('lower', 'value'),State('upper', 'value'),State('TYPE','value')])

def update_fft(n_clicks,ch_id,selection,value,toggle,lower,upper,type):

    path0 = value['props']
    path0 = json.loads(path0['children'])
    ch_id = ch_id
    if ch_id is None or 'Filename' not in path0.keys() or check_filename(path0['Filename']) == False or 'xaxis.range[0]' not in selection or 'xaxis.range[1]' not in selection:
        return fig
    else:
        x0 = selection['xaxis.range[0]']
        x1 = selection['xaxis.range[1]']
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        tot_chs = parameters['numRecElectrodes']
        Frames = int(path0['Frames'])
        ch_x = np.linspace(0, Frames, Frames) / parameters['samplingRate']
        range_lower = np.where(ch_x >= x0)[0]
        range_upper = np.where(ch_x >= x1)[0]
        fig5 = make_subplots(rows=len(ch_id), cols=1, shared_xaxes=True, vertical_spacing=0.06, x_title="Frequency, Hz", y_title="Voltage, (mV)")

        plot = 0
        if len(ch_id)<2:
            width_plot = 500
        else:
            width_plot = 300


        for item in ch_id:
            row, column = get_row_col_num(int(item))
            ch_id1 = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
            data = np.array(h5['/3BData/Raw']).reshape(Frames, tot_chs)
            ch_y = data[:, ch_id1]
            CH_Y = convert_to_uV(ch_y, parameters) / 1000000
            CH_Y = (CH_Y - np.mean(CH_Y))
            sig = CH_Y[range_lower[0]:range_upper[0]]
            freq, sig_fft = fft(sig, parameters['samplingRate'])

            label = ('(' + str(row) + ',' + str(column) + ')')
            if toggle == False:
                fig5.add_trace(go.Scatter(x=freq, y=sig_fft, mode='lines', name=label), row=plot+1, col=1)
                fig5.update_layout(height=len(ch_id) * width_plot, width=1000, title_text="Spectrum")
                fig5.update_xaxes(showline=True, linewidth=1, linecolor='black', type = 'log',mirror=True, showticklabels=True)
            else:
                CH_Y_f = frequency_filter(CH_Y[range_lower[0]:range_upper[0]],parameters['samplingRate'], type, int(lower), int(upper), order=6)
                sig_f = CH_Y_f
                freq_f, sig_fft_f = fft(sig_f, parameters['samplingRate'])
                fig5.add_trace(go.Scatter(x=freq, y=sig_fft, mode='lines', name=label), row=plot+1, col=1)
                fig5.add_trace( go.Scatter(x=freq_f, y=sig_fft_f, mode='lines', name=label + 'filter'),row= plot+1, col=1)
                fig5.update_layout(height=len(ch_id)*width_plot, width=1000, title_text="Spectrum - Filtered")
                fig5.update_xaxes(showline=True, linewidth=1, linecolor='black', type = 'log',mirror=True, showticklabels=True)
            plot+=1

        fig5.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig5.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig5.update_layout(template="plotly_white", showlegend=True, legend=dict(orientation="h"))

        h5.close()

        return fig5

@app.callback(
    [Output('filt-g1', 'figure'),Output('table-sz-gp1','data')],
    [Input('ch_list-sz-g1', 'value'), Input('file_name_text', 'children'), Input('window1-g1', 'value'),
     Input('cutoff1-g1', 'value'), Input('window2-g1', 'value'), Input('cutoff2-g1', 'value'),
     Input('detect_mode', 'value')])
def update_figure(ch_num, value, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    ch_id = ch_num
    if ch_id is None or 'Filename' not in path0.keys() or check_filename(path0['Filename']) == False:
        return fig0, []
    else:
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        Frames = parameters['nRecFrames']
        sampling = parameters['samplingRate']
        chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
        row, column = get_row_col_num(ch_id)
        ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
        x = data[:, ch_id]
        x = convert_to_uV(x, parameters) / 1000000
        x = x - np.mean(x)
        x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
        t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,int(smooth1),float(cutoff1), int(smooth2), float(cutoff2),0.018,25)
        df_loop = detect_seizures(events, events_pp, tt, t)
        df_loop = check_false_positive(df_loop, peaks_raster,sampling, Frames)
        env_time = np.linspace(0, int(t[-1]), int(t[-1]))
        env = np.zeros(len(env_time), dtype=int)
        annot_time = []
        annot_text = []
        annot_value = []

        starts = []
        ends = []
        sz_nm = []
        count = 0
        for i in df_loop.index:
            if df_loop.loc[i, 'type'] != 'NS' and (~np.isnan(df_loop.loc[i,'start'])):

                s = df_loop.loc[i, 'start']
                e = df_loop.loc[i, 'end']
                starts.append(s)
                ends.append(e)
                sz_nm.append(count)
                num = count+1
                text = df_loop.loc[i, 'type']
                env[s:e] = 1
                annot_time.append((s + e) / 2)
                annot_value.append(1)
                annot_text.append(text[0]+"("+str(num)+")")
                count+=1

        df_IEF = get_ief(peaks_raster,sampling,Frames,starts,ends,sz_nm)       
        fig3 = go.Figure(data=go.Scatter(x=t[10000:], y=x[10000:], marker_color="green", name='Time Trace'))
        fig3.add_trace(go.Scatter(x=env_time, y=env, mode='lines', name='envelop'))
        fig3.add_trace(go.Scatter(x=annot_time, y=annot_value, mode='markers+text', text=annot_text, textposition="top center", name='events'))
        fig3.update_xaxes(showline=True, linewidth=1, title='Time, seconds', linecolor='black', mirror=True)
        fig3.update_yaxes(showline=True, linewidth=1, title='Voltage, microVolts', linecolor='black', mirror=True)
        fig3.update_layout(template="plotly_white", showlegend=True, height=800, width=1000, legend=dict(orientation="h"))
        h5.close()

        return fig3, df_IEF


@app.callback(
    [Output('path-g1', 'figure'), Output('table4', 'data')],
    [Input('sz-analysis', 'value'), Input('g2_g1', 'selectedData'), Input('file_name_text', 'children'),
     Input('g7', 'relayoutData'), Input('window1-g1', 'value'), Input('cutoff1-g1', 'value'),
     Input('window2-g1', 'value'), Input('cutoff2-g1', 'value'), Input('detect_mode', 'value')])
def display_selected_ch(tab, selectedData, value, selection, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    points = selectedData
    path0 = value['props']
    path0 = json.loads(path0['children'])
    if tab == 'sz-plot':
        if points is None or selection is None or 'xaxis.range[0]' not in selection or 'xaxis.range[1]' not in selection:
            default = []
            return fig0, []
        else:
            x0 = selection['xaxis.range[0]']
            x1 = selection['xaxis.range[1]']
            path = path0['Filename']
            h5 = h5py.File(path, 'r')
            parameters = parameter(h5)
            chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
            Frames = parameters['nRecFrames']
            sampling = parameters['samplingRate']
            chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
            data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
            df_summary = pd.DataFrame(columns=['Channel #', 'Row', 'Column', 'Seizure-Count', 'Mean-Duration'])
            df_rank_sz = pd.DataFrame(columns=['channel', 'id', 'start', 'end'])
            row_list, column_list, row_list_noise, column_list_noise = get_row_column_list(data, chsList, parameters)
            count = 0
            group_row = []
            group_column = []
            table_dict = {}
            lfp_sub = []
            ch_list = []
            ttf_event = []
            df_channel = pd.DataFrame(columns=['ch_num', 'lfp_count', 'ttfe'])

            for item in points["points"]:
                sz_rank_df = pd.DataFrame()
                row = item['y']
                column = item['x']
                group_row.append(row)
                group_column.append(column)
                ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
                x = data[:, ch_id]
                x = convert_to_uV(x, parameters) / 100000
                x = x - np.mean(x)
                x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
                t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,int(smooth1),float(cutoff1), int(smooth2),float(cutoff2), 0.018, 25)
                df_SZ = detect_seizures(events, events_pp, tt, t)
                df_SZ = check_false_positive(df_SZ, peaks_raster,sampling, Frames)
                sz_rank_df = df_SZ[['id', 'start', 'end']].copy()
                sz_rank_df.reset_index()
                reset_x = 0
                sz_rank_df.dropna()

                for i in sz_rank_df.index:
                    sz_rank_df.loc[i, 'id'] = reset_x
                    reset_x += 1

                if len(sz_rank_df) > 0 and (~np.isnan(sz_rank_df.loc[sz_rank_df.index[0], 'start'])):
                    ss = sz_rank_df.loc[sz_rank_df.index[0], 'start']
                else:
                    ss = x1

                ch_list.append(ch_id)
                lfp_sub.append(np.sum(peaks_raster[int(x0 * sampling):int(ss * sampling)]))
                ttt = np.where(peaks_raster > 0)

                ttt = ttt[0] / sampling
                ttfe = []
                for i in ttt:
                    if i > x0 and i < x1:
                        ttfe.append(i)

                if len(ttfe) > 0:
                    ttf_event.append(ttfe[0])
                else:
                    ttf_event.append(1000000000)

                sz_rank_df['channel'] = get_ch_number(item['y'], item['x'])
                sz_rank_df['row'] = item['y']
                sz_rank_df['column'] = item['x']
                sz_rank_df.dropna()
                df_rank_sz = df_rank_sz.append(sz_rank_df, ignore_index=True)
                seizures = len(df_SZ[df_SZ['type'] == 'SZ'])
                mean_duration = df_SZ['duration'].mean()
                df_summary.loc[count, 'Channel #'] = get_ch_number(item['y'], item['x'])
                df_summary.loc[count, 'Row'] = row
                df_summary.loc[count, 'Column'] = column
                df_summary.loc[count, 'Seizure-Count'] = seizures
                df_summary.loc[count, 'Mean-Duration'] = round(mean_duration, 0)
                count += 1

            df_channel['ch_num'] = ch_list
            df_channel['lfp_count'] = lfp_sub
            df_channel['ttfe'] = ttf_event
            groups = {}

            groups['Time-Window'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
            groups['Group'] = 1
            groups['Tot-Channel'] = len(df_channel)
            df_channel = df_channel[df_channel['lfp_count'] > 0]

            cut_off_lfp = np.mean(np.array(df_channel['lfp_count'])) + 5 * np.std(np.array(df_channel['lfp_count']))
            df_channel = df_channel[df_channel['lfp_count'] < cut_off_lfp]
            groups['LFP-Count'] = np.sum(np.array(df_channel['lfp_count']))
            groups['Act-Channel'] = len(df_channel)
            groups['LFP-Count-perCH'] = round(np.sum(np.array(df_channel['lfp_count'])) / len(df_channel), 0)
            groups['LFP-Count-per-Time'] = round(np.sum(np.array(df_channel['lfp_count'])) / (x1 - x0) / len(df_channel), 2)

            if len(list(df_channel['ttfe'])) > 0:
                groups['time-first-event'] = min(list(df_channel['ttfe']))
            else:
                groups['time-first-event'] = 0

            table3 = df_summary.to_dict("records")
            df_rank_sz = df_rank_sz.sort_values(by='start', ascending=True)
            df_rank_sz['duration'] = df_rank_sz['end'] - df_rank_sz['start']
            df_rank_sz = df_rank_sz[df_rank_sz['end']//100 <= x1//100]
            df_rank_sz = df_rank_sz[df_rank_sz['start']//100 >= x0//100]
            
            if len(df_rank_sz) > 0:
                initial_time = list(df_rank_sz['start'])[0]
                row_ss = list(df_rank_sz['row'])[0]
                column_ss = list(df_rank_sz['column'])[0]
                df_rank_sz['distance'] = ((df_rank_sz['row'] - row_ss) ** 2 + (
                            df_rank_sz['column'] - column_ss) ** 2) ** 0.5
                table_dict['Distance'] = round(max(list(df_rank_sz['distance'])), 2)
                table_dict['Max-Duration'] = round(max(list(df_rank_sz['duration'])), 2)
                table_dict['Mean-Duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                table_dict['time-int'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
                df_rank_sz['tt_sz'] = df_rank_sz['start'].shift(1)
                df_rank_sz['tt_sz'].iloc[0] = df_rank_sz['start'].iloc[0]
                df_rank_sz['tt_sz'] = df_rank_sz['start'] - df_rank_sz['tt_sz']
                rec_row = list(df_rank_sz['row'])
                rec_column = list(df_rank_sz['column'])
                channels_start = df_rank_sz[df_rank_sz['start'] == initial_time]['channel']
                duration = max(list(df_rank_sz[df_rank_sz['start'] == initial_time]['end'])) - initial_time
                table_dict['sz-rate'] = round(table_dict['Distance'] / (np.mean(list(df_rank_sz['tt_sz']))), 2)
                groups['SZ-Channels'] = len(rec_row)
                groups['SZ-max-duration'] = round(max(list(df_rank_sz['duration'])), 2)
                groups['SZ-mean-duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                groups['SZ-distance'] = round(max(list(df_rank_sz['distance'])), 2)
                groups['SZ-rate'] = round(table_dict['Distance'] / (np.mean(list(df_rank_sz['tt_sz']))), 2)

                start_row = []
                start_column = []
                for iterr in channels_start:
                    row_ref, column_ref = get_row_col_num(iterr)
                    start_row.append(row_ref)
                    start_column.append(column_ref)

                groups['SZ-start'] = str(start_row[0]) + ', ' + str(start_column[0])
            else:
                table_dict['Distance'] = 0
                table_dict['Max-Duration'] = 0
                table_dict['Mean-Duration'] = 0
                table_dict['time-int'] = 0
                table_dict['sz-rate'] = 0

                groups['SZ-Channels'] = 0
                groups['SZ-max-duration'] = 0
                groups['SZ-mean-duration'] = 0
                groups['SZ-distance'] = 0
                groups['SZ-rate'] = 0
                groups['SZ-start'] = str(0) + ', ' + str(0)
                rec_row = []
                rec_column = []
                channels_start = []
                start_column = []
                start_row = []

            groups['Time-Stamp'] = str(datetime.now())
            output = '\\'.join(path0['Filename'].split('\\')[0:-1]) + '\\results-' + path0['Filename'].split('\\')[-1].split('.')[0]
            csv_file_name = output + '\\' + 'group_summary_log.csv'

            with open(csv_file_name, 'a') as myfile:
                writer = csv.DictWriter(myfile,fieldnames=['Time-Stamp', 'Time-Window', 'Group', 'LFP-Count', 'Tot-Channel',
                                                    'Act-Channel', 'LFP-Count-perCH', 'LFP-Count-per-Time',
                                                    'time-first-event', 'SZ-Channels', 'SZ-start', 'SZ-max-duration','SZ-mean-duration',
                                                    'SZ-distance', 'SZ-rate'])
                writer.writerow(groups)
                myfile.close()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=column_list, y=row_list, marker={'color': 'grey', 'showscale': False}, mode='markers', name='All Active Channels'))
            fig2.add_trace(go.Scatter(x=group_column, y=group_row, marker={'color': 'blue', 'showscale': False}, mode='markers', name='Group1 Channels'))
            fig2.add_trace(go.Scatter(x=rec_column, y=rec_row, marker={'color': 'red', 'showscale': False}, mode='markers', name='Channels Recruited'))
            fig2.add_trace(go.Scatter(x=start_column, y=start_row, marker={'color': 'green', 'showscale': False}, mode='markers', name='Point of Initiation'))
            fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
            fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
            fig2.update_layout(template="plotly_white", clickmode='event+select', width=800, height=800, legend=dict(orientation="h"))
            return fig2, [table_dict,]
    else:
        return fig0, []


@app.callback(
    [Output('filt-g2', 'figure'), Output('table-sz-gp2','data')],
    [Input('ch_list-sz-g2', 'value'), Input('file_name_text', 'children'), Input('window1-g2', 'value'),
     Input('cutoff1-g2', 'value'), Input('window2-g2', 'value'), Input('cutoff2-g2', 'value'),
     Input('detect_mode', 'value')])
def update_figure(ch_num, value, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    ch_id = ch_num
    if ch_id is None or 'Filename' not in path0.keys() or check_filename(path0['Filename']) == False:
        return fig0, []
    else:
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        Frames = parameters['nRecFrames']
        sampling = parameters['samplingRate']
        chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
        row, column = get_row_col_num(ch_id)
        ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
        x = data[:, ch_id]
        x = convert_to_uV(x, parameters) / 1000000
        x = x - np.mean(x)
        x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
        t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,int(smooth1), float(cutoff1), int(smooth2), float(cutoff2),0.018,25)
        df_loop = detect_seizures(events, events_pp, tt, t)
        df_loop = check_false_positive(df_loop, peaks_raster,sampling, Frames)
        env_time = np.linspace(0, int(t[-1]), int(t[-1]))
        env = np.zeros(len(env_time), dtype=int)
        annot_time = []
        annot_text = []
        annot_value = []
        starts = []
        ends = []
        sz_nm = []
        count = 0
        for i in df_loop.index:
            if df_loop.loc[i, 'type'] != 'NS' and (~np.isnan(df_loop.loc[i,'start'])):
                s = df_loop.loc[i, 'start']
                e = df_loop.loc[i, 'end']
                starts.append(s)
                ends.append(e)
                sz_nm.append(count)
                num = count+1
                text = df_loop.loc[i, 'type']
                env[s:e] = 1
                annot_time.append((s + e) / 2)
                annot_value.append(1)
                annot_text.append(text[0]+"("+str(num)+")")
                count+=1

        df_IEF = get_ief(peaks_raster,sampling,Frames,starts,ends,sz_nm)
        fig3 = go.Figure(data=go.Scatter(x=t[10000:], y=x[10000:], marker_color="green", name='Time Trace'))
        fig3.add_trace(go.Scatter(x=env_time, y=env, mode='lines', name='envelop'))
        fig3.add_trace(go.Scatter(x=annot_time, y=annot_value, mode='markers+text', text=annot_text, textposition="top center", name='events'))
        fig3.update_xaxes(showline=True, linewidth=1, title='Time, seconds', linecolor='black', mirror=True)
        fig3.update_yaxes(showline=True, linewidth=1, title='Voltage, microVolts', linecolor='black', mirror=True)
        fig3.update_layout(template="plotly_white", showlegend=True, height=800, width=1000, legend=dict(orientation="h"))
        h5.close()

        return fig3, df_IEF


@app.callback(
    [Output('path-g2', 'figure'), Output('table5', 'data')],
    [Input('sz-analysis', 'value'), Input('g2_g2', 'selectedData'), Input('file_name_text', 'children'),
     Input('g7', 'relayoutData'), Input('window1-g2', 'value'), Input('cutoff1-g2', 'value'),
     Input('window2-g2', 'value'), Input('cutoff2-g2', 'value'), Input('detect_mode', 'value')])
def display_selected_ch(tab, selectedData, value, selection, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    points = selectedData
    path0 = value['props']
    path0 = json.loads(path0['children'])

    if tab == 'sz-plot':
        if points is None or selection is None or 'xaxis.range[0]' not in selection or 'xaxis.range[1]' not in selection:
            default = []
            return fig0, []
        else:
            x0 = selection['xaxis.range[0]']
            x1 = selection['xaxis.range[1]']
            path = path0['Filename']
            h5 = h5py.File(path, 'r')
            parameters = parameter(h5)
            chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
            Frames = parameters['nRecFrames']
            sampling = parameters['samplingRate']
            chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
            data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
            df_summary = pd.DataFrame(columns=['Channel #', 'Row', 'Column', 'Seizure-Count', 'Mean-Duration'])
            df_rank_sz = pd.DataFrame(columns=['channel', 'id', 'start', 'end'])
            row_list, column_list, row_list_noise, column_list_noise = get_row_column_list(data, chsList, parameters)
            count = 0
            group_row = []
            group_column = []
            table_dict = {}
            lfp_sub = []
            ch_list = []
            ttf_event = []
            df_channel = pd.DataFrame(columns=['ch_num', 'lfp_count', 'ttfe'])

            for item in points["points"]:
                sz_rank_df = pd.DataFrame()
                row = item['y']
                column = item['x']
                group_row.append(row)
                group_column.append(column)
                ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
                x = data[:, ch_id]
                x = convert_to_uV(x, parameters) / 1000000
                x = x - np.mean(x)
                x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
                t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,
                                                                             int(smooth1),
                                                                             float(cutoff1), int(smooth2),
                                                                             float(cutoff2), 0.018, 25)
                df_SZ = detect_seizures(events, events_pp, tt, t)
                df_SZ = check_false_positive(df_SZ, peaks_raster,sampling, Frames)

                sz_rank_df = df_SZ[['id', 'start', 'end']].copy()
                sz_rank_df.reset_index()
                reset_x = 0
                sz_rank_df.dropna()

                for i in sz_rank_df.index:
                    sz_rank_df.loc[i, 'id'] = reset_x
                    reset_x += 1

                if len(sz_rank_df) > 0 and (~np.isnan(sz_rank_df.loc[sz_rank_df.index[0], 'start'])):
                    ss = sz_rank_df.loc[sz_rank_df.index[0], 'start']
                else:
                    ss = x1

                ch_list.append(ch_id)
                lfp_sub.append(np.sum(peaks_raster[int(x0 * sampling):int(ss * sampling)]))
                ttt = np.where(peaks_raster > 0)

                ttt = ttt[0] / sampling
                ttfe = []
                for i in ttt:
                    if i > x0 and i < x1:
                        ttfe.append(i)

                if len(ttfe) > 0:
                    ttf_event.append(ttfe[0])
                else:
                    ttf_event.append(1000000000)

                sz_rank_df['channel'] = get_ch_number(item['y'], item['x'])
                sz_rank_df['row'] = item['y']
                sz_rank_df['column'] = item['x']
                sz_rank_df.dropna()
                df_rank_sz = df_rank_sz.append(sz_rank_df, ignore_index=True)
                seizures = len(df_SZ[df_SZ['type'] == 'SZ'])
                mean_duration = df_SZ['duration'].mean()
                df_summary.loc[count, 'Channel #'] = get_ch_number(item['y'], item['x'])
                df_summary.loc[count, 'Row'] = row
                df_summary.loc[count, 'Column'] = column
                df_summary.loc[count, 'Seizure-Count'] = seizures
                df_summary.loc[count, 'Mean-Duration'] = round(mean_duration, 0)
                count += 1

            df_channel['ch_num'] = ch_list
            df_channel['lfp_count'] = lfp_sub
            df_channel['ttfe'] = ttf_event
 
            groups = {}
            groups['Time-Window'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
            groups['Group'] = 2
            groups['Tot-Channel'] = len(df_channel)
            df_channel = df_channel[df_channel['lfp_count'] > 0]
            cut_off_lfp = np.mean(np.array(df_channel['lfp_count'])) + 5 * np.std(np.array(df_channel['lfp_count']))
            df_channel = df_channel[df_channel['lfp_count'] < cut_off_lfp]
            groups['LFP-Count'] = np.sum(np.array(df_channel['lfp_count']))
            groups['Act-Channel'] = len(df_channel)
            groups['LFP-Count-perCH'] = round(np.sum(np.array(df_channel['lfp_count'])) / len(df_channel), 0)
            groups['LFP-Count-per-Time'] = round(np.sum(np.array(df_channel['lfp_count'])) / (x1 - x0) / len(df_channel), 2)
            if len(list(df_channel['ttfe'])) > 0:
                groups['time-first-event'] = min(list(df_channel['ttfe']))
            else:
                groups['time-first-event'] = 0

            table3 = df_summary.to_dict("records")
            df_rank_sz = df_rank_sz.sort_values(by='start', ascending=True)
            df_rank_sz['duration'] = df_rank_sz['end'] - df_rank_sz['start']
            df_rank_sz = df_rank_sz[df_rank_sz['end'] <= x1]
            df_rank_sz = df_rank_sz[df_rank_sz['start'] >= x0]

            if len(df_rank_sz) > 0:
                initial_time = list(df_rank_sz['start'])[0]
                row_ss = list(df_rank_sz['row'])[0]
                column_ss = list(df_rank_sz['column'])[0]
                df_rank_sz['distance'] = ((df_rank_sz['row'] - row_ss) ** 2 + (df_rank_sz['column'] - column_ss) ** 2) ** 0.5
                table_dict['Distance'] = round(max(list(df_rank_sz['distance'])), 2)
                table_dict['Max-Duration'] = round(max(list(df_rank_sz['duration'])), 2)
                table_dict['Mean-Duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                table_dict['time-int'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
                df_rank_sz['tt_sz'] = df_rank_sz['start'].shift(1)
                df_rank_sz['tt_sz'].iloc[0] = df_rank_sz['start'].iloc[0]
                df_rank_sz['tt_sz'] = df_rank_sz['start'] - df_rank_sz['tt_sz']
                rec_row = list(df_rank_sz['row'])
                rec_column = list(df_rank_sz['column'])
                channels_start = df_rank_sz[df_rank_sz['start'] == initial_time]['channel']
                duration = max(list(df_rank_sz[df_rank_sz['start'] == initial_time]['end'])) - initial_time
                table_dict['sz-rate'] = round(table_dict['Distance'] / (np.mean(list(df_rank_sz['tt_sz']))), 2)
                groups['SZ-Channels'] = len(rec_row)
                groups['SZ-max-duration'] = round(max(list(df_rank_sz['duration'])), 2)
                groups['SZ-mean-duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                groups['SZ-distance'] = round(max(list(df_rank_sz['distance'])), 2)
                groups['SZ-rate'] = round(table_dict['Distance'] / (np.mean(list(df_rank_sz['tt_sz']))), 2)
                start_row = []
                start_column = []
                for iterr in channels_start:
                    row_ref, column_ref = get_row_col_num(iterr)
                    start_row.append(row_ref)
                    start_column.append(column_ref)
                groups['SZ-start'] = str(start_row[0]) + ', ' + str(start_column[0])
            else:
                table_dict['Distance'] = 0
                table_dict['Max-Duration'] = 0
                table_dict['Mean-Duration'] = 0
                table_dict['time-int'] = 0
                table_dict['sz-rate'] = 0
                groups['SZ-Channels'] = 0
                groups['SZ-max-duration'] = 0
                groups['SZ-max-duration'] = 0
                groups['SZ-distance'] = 0
                groups['SZ-rate'] = 0
                groups['SZ-start'] = str(0) + ', ' + str(0)
                rec_row = []
                rec_column = []
                channels_start = []
                start_column = []
                start_row = []

            groups['Time-Stamp'] = str(datetime.now())
            output = '\\'.join(path0['Filename'].split('\\')[0:-1]) + '\\results-' + path0['Filename'].split('\\')[-1].split('.')[0]
            csv_file_name = output + '\\' + 'group_summary_log.csv'

            with open(csv_file_name, 'a') as myfile:
                writer = csv.DictWriter(myfile,
                                        fieldnames=['Time-Stamp', 'Time-Window', 'Group', 'LFP-Count', 'Tot-Channel',
                                                    'Act-Channel', 'LFP-Count-perCH', 'LFP-Count-per-Time',
                                                    'time-first-event', 'SZ-Channels', 'SZ-start', 'SZ-max-duration','SZ-mean-duration',
                                                    'SZ-distance', 'SZ-rate'])
                writer.writerow(groups)
                myfile.close()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=column_list, y=row_list, marker={'color': 'grey', 'showscale': False}, mode='markers', name='All Active Channels'))
            fig2.add_trace(go.Scatter(x=group_column, y=group_row, marker={'color': 'blue', 'showscale': False}, mode='markers', name='Group1 Channels'))
            fig2.add_trace(go.Scatter(x=rec_column, y=rec_row, marker={'color': 'red', 'showscale': False}, mode='markers', name='Channels Recruited'))
            fig2.add_trace(go.Scatter(x=start_column, y=start_row, marker={'color': 'green', 'showscale': False}, mode='markers', name='Point of Initiation'))
            fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
            fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
            fig2.update_layout(template="plotly_white", clickmode='event+select', width=800, height=800, legend=dict(orientation="h"))

            return fig2, [table_dict, ]
    else:
        return fig0, []


@app.callback(
    [Output('filt-g3', 'figure'),Output('table-sz-gp3','data')],
    [Input('ch_list-sz-g3', 'value'), Input('file_name_text', 'children'), Input('window1-g3', 'value'),
     Input('cutoff1-g3', 'value'), Input('window2-g3', 'value'), Input('cutoff2-g3', 'value'),
     Input('detect_mode', 'value')])
def update_figure(ch_num, value, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    path0 = value['props']
    path0 = json.loads(path0['children'])
    ch_id = ch_num
    if ch_id is None or 'Filename' not in path0.keys() or check_filename(path0['Filename']) == False:
        return fig0, []
    else:
        path = path0['Filename']
        h5 = h5py.File(path, 'r')
        parameters = parameter(h5)
        chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        Frames = parameters['nRecFrames']
        sampling = parameters['samplingRate']
        chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
        data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
        row, column = get_row_col_num(ch_id)
        ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
        x = data[:, ch_id]
        x = convert_to_uV(x, parameters) / 100000
        x = x - np.mean(x)
        x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
        t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,int(smooth1),float(cutoff1), int(smooth2), float(cutoff2),0.018,25)
        df_loop = detect_seizures(events, events_pp, tt, t)
        df_loop = check_false_positive(df_loop, peaks_raster,sampling, Frames)
        env_time = np.linspace(0, int(t[-1]), int(t[-1]))
        env = np.zeros(len(env_time), dtype=int)
        annot_time = []
        annot_text = []
        annot_value = []
        starts = []
        ends = []
        sz_nm = []
        count = 0
        for i in df_loop.index:
            if df_loop.loc[i, 'type'] != 'NS' and (~np.isnan(df_loop.loc[i,'start'])):
                s = df_loop.loc[i, 'start']
                e = df_loop.loc[i, 'end']
                starts.append(s)
                ends.append(e)
                sz_nm.append(count)
                num = count+1
                text = df_loop.loc[i, 'type']
                env[s:e] = 1
                annot_time.append((s + e) / 2)
                annot_value.append(1)
                annot_text.append(text[0]+"("+str(num)+")")
                count+=1
        df_IEF = get_ief(peaks_raster,sampling,Frames,starts,ends,sz_nm)
        fig3 = go.Figure(data=go.Scatter(x=t[10000:], y=x[10000:], marker_color="green", name='Time Trace'))
        fig3.add_trace(go.Scatter(x=env_time, y=env, mode='lines', name='envelop'))
        fig3.add_trace(go.Scatter(x=annot_time, y=annot_value, mode='markers+text', text=annot_text, textposition="top center", name='events'))
        fig3.update_xaxes(showline=True, linewidth=1, title='Time, seconds', linecolor='black', mirror=True)
        fig3.update_yaxes(showline=True, linewidth=1, title='Voltage, microVolts', linecolor='black', mirror=True)
        fig3.update_layout(template="plotly_white", showlegend=True, height=800, width=1000, legend=dict(orientation="h"))
        h5.close()

        return fig3, df_IEF


@app.callback(
    [Output('path-g3', 'figure'), Output('table6', 'data')],
    [Input('sz-analysis', 'value'), Input('g2_g3', 'selectedData'), Input('file_name_text', 'children'),
     Input('g7', 'relayoutData'), Input('window1-g3', 'value'), Input('cutoff1-g3', 'value'),
     Input('window2-g3', 'value'), Input('cutoff2-g3', 'value'), Input('detect_mode', 'value')])
def display_selected_ch(tab, selectedData, value, selection, smooth1, cutoff1, smooth2, cutoff2, detect_mode):
    points = selectedData
    path0 = value['props']
    path0 = json.loads(path0['children'])
    if tab == 'sz-plot':
        if points is None or selection is None or 'xaxis.range[0]' not in selection or 'xaxis.range[1]' not in selection:
            default = []
            return fig0, []
        else:
            x0 = selection['xaxis.range[0]']
            x1 = selection['xaxis.range[1]']
            path = path0['Filename']
            h5 = h5py.File(path, 'r')
            parameters = parameter(h5)
            chsList = h5['/3BRecInfo/3BMeaStreams/Raw/Chs'][:]
            Frames = parameters['nRecFrames']
            sampling = parameters['samplingRate']
            chs = h5['/3BRecInfo/3BMeaStreams/Raw/Chs']
            data = np.array(h5['/3BData/Raw']).reshape(Frames, len(chs[:]))
            df_summary = pd.DataFrame(columns=['Channel #', 'Row', 'Column', 'Seizure-Count', 'Mean-Duration'])
            df_rank_sz = pd.DataFrame(columns=['channel', 'id', 'start', 'end'])
            row_list, column_list, row_list_noise, column_list_noise = get_row_column_list(data, chsList, parameters)
            count = 0
            group_row = []
            group_column = []
            table_dict = {}
            lfp_sub = []
            ch_list = []
            ttf_event = []
            df_channel = pd.DataFrame(columns=['ch_num', 'lfp_count', 'ttfe'])

            for item in points["points"]:
                sz_rank_df = pd.DataFrame()
                row = item['y']
                column = item['x']
                group_row.append(row)
                group_column.append(column)
                ch_id = np.where((chsList['Row'] == row) & (chsList['Col'] == column))[0][0]
                x = data[:, ch_id]
                x = convert_to_uV(x, parameters) / 100000
                x = x - np.mean(x)
                x = frequency_filter(x, sampling, "BTR", int(0), int(15), order=6)
                t, events, tt, events_pp, peaks_raster = get_events_envelope(x, sampling, Frames, detect_mode,int(smooth1),float(cutoff1), int(smooth2),float(cutoff2), 0.018, 25)
                df_SZ = detect_seizures(events, events_pp, tt, t)
                df_SZ = check_false_positive(df_SZ, peaks_raster,sampling, Frames)
                sz_rank_df = df_SZ[['id', 'start', 'end']].copy()
                sz_rank_df.reset_index()
                reset_x = 0
                sz_rank_df.dropna()

                for i in sz_rank_df.index:
                    sz_rank_df.loc[i, 'id'] = reset_x
                    reset_x += 1

                if len(sz_rank_df) > 0 and (~np.isnan(sz_rank_df.loc[sz_rank_df.index[0], 'start'])):
                    ss = sz_rank_df.loc[sz_rank_df.index[0], 'start']
                else:
                    ss = x1

                ch_list.append(ch_id)
                lfp_sub.append(np.sum(peaks_raster[int(x0 * sampling):int(ss * sampling)]))
                ttt = np.where(peaks_raster > 0)
                ttt = ttt[0] / sampling
                ttfe = []
                for i in ttt:
                    if i > x0 and i < x1:
                        ttfe.append(i)

                if len(ttfe) > 0:
                    ttf_event.append(ttfe[0])
                else:
                    ttf_event.append(1000000000)

                sz_rank_df['channel'] = get_ch_number(item['y'], item['x'])
                sz_rank_df['row'] = item['y']
                sz_rank_df['column'] = item['x']
                sz_rank_df.dropna()
                df_rank_sz = df_rank_sz.append(sz_rank_df, ignore_index=True)
                seizures = len(df_SZ[df_SZ['type'] == 'SZ'])
                mean_duration = df_SZ['duration'].mean()
                df_summary.loc[count, 'Channel #'] = get_ch_number(item['y'], item['x'])
                df_summary.loc[count, 'Row'] = row
                df_summary.loc[count, 'Column'] = column
                df_summary.loc[count, 'Seizure-Count'] = seizures
                df_summary.loc[count, 'Mean-Duration'] = round(mean_duration, 0)
                count += 1

            df_channel['ch_num'] = ch_list
            df_channel['lfp_count'] = lfp_sub
            df_channel['ttfe'] = ttf_event
            groups = {}
            groups['Time-Window'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
            groups['Group'] = 3
            groups['Tot-Channel'] = len(df_channel)
            df_channel = df_channel[df_channel['lfp_count'] > 0]
            cut_off_lfp = np.mean(np.array(df_channel['lfp_count'])) + 5 * np.std(np.array(df_channel['lfp_count']))
            df_channel = df_channel[df_channel['lfp_count'] < cut_off_lfp]
            groups['LFP-Count'] = np.sum(np.array(df_channel['lfp_count']))
            groups['Act-Channel'] = len(df_channel)
            groups['LFP-Count-perCH'] = round(np.sum(np.array(df_channel['lfp_count'])) / len(df_channel), 0)
            groups['LFP-Count-per-Time'] = round(np.sum(np.array(df_channel['lfp_count'])) / (x1 - x0) / len(df_channel), 2)
            if len(list(df_channel['ttfe'])) > 0:
                groups['time-first-event'] = min(list(df_channel['ttfe']))
            else:
                groups['time-first-event'] = 0
            table3 = df_summary.to_dict("records")
            df_rank_sz = df_rank_sz.sort_values(by='start', ascending=True)
            df_rank_sz['duration'] = df_rank_sz['end'] - df_rank_sz['start']
            df_rank_sz = df_rank_sz[df_rank_sz['end'] <= x1]
            df_rank_sz = df_rank_sz[df_rank_sz['start'] >= x0]
            if len(df_rank_sz) > 0:
                initial_time = list(df_rank_sz['start'])[0]
                row_ss = list(df_rank_sz['row'])[0]
                column_ss = list(df_rank_sz['column'])[0]
                df_rank_sz['distance'] = ((df_rank_sz['row'] - row_ss) ** 2 + (df_rank_sz['column'] - column_ss) ** 2) ** 0.5
                table_dict['Distance'] = round(max(list(df_rank_sz['distance'])), 2)
                table_dict['Max-Duration'] = round(max(list(df_rank_sz['duration'])), 2)
                table_dict['Mean-Duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                table_dict['time-int'] = str(round(x0, 0)) + " to " + str(round(x1, 0))
                df_rank_sz['tt_sz'] = df_rank_sz['start'].shift(1)
                df_rank_sz['tt_sz'].iloc[0] = df_rank_sz['start'].iloc[0]
                df_rank_sz['tt_sz'] = df_rank_sz['start'] - df_rank_sz['tt_sz']
                rec_row = list(df_rank_sz['row'])
                rec_column = list(df_rank_sz['column'])
                channels_start = df_rank_sz[df_rank_sz['start'] == initial_time]['channel']
                duration = max(list(df_rank_sz[df_rank_sz['start'] == initial_time]['end'])) - initial_time
                table_dict['sz-rate'] = round(table_dict['Distance'] / (np.mean(list(df_rank_sz['tt_sz']))), 2)
                groups['SZ-Channels'] = len(rec_row)
                groups['SZ-max-duration'] = round(max(list(df_rank_sz['duration'])), 2)
                groups['SZ-mean-duration'] = round(np.mean(np.array(df_rank_sz['duration'])), 2)
                groups['SZ-distance'] = round(max(list(df_rank_sz['distance'])), 2)
                groups['SZ-rate'] = round(table_dict['Distance'] / (np.mean(np.array(df_rank_sz['tt_sz']))), 2)
                start_row = []
                start_column = []
                for iterr in channels_start:
                    row_ref, column_ref = get_row_col_num(iterr)
                    start_row.append(row_ref)
                    start_column.append(column_ref)
                groups['SZ-start'] = str(start_row[0]) + ', ' + str(start_column[0])
            else:
                table_dict['Distance'] = 0
                table_dict['Max-Duration'] = 0
                table_dict['Mean-Duration'] = 0
                table_dict['time-int'] = 0
                table_dict['sz-rate'] = 0
                groups['SZ-Channels'] = 0
                groups['SZ-max-duration'] = 0
                groups['SZ-mean-duration'] = 0
                groups['SZ-distance'] = 0
                groups['SZ-rate'] = 0
                groups['SZ-start'] = str(0) + ', ' + str(0)
                rec_row = []
                rec_column = []
                channels_start = []
                start_column = []
                start_row = []

            groups['Time-Stamp'] = str(datetime.now())
            output = '\\'.join(path0['Filename'].split('\\')[0:-1]) + '\\results-' + path0['Filename'].split('\\')[-1].split('.')[0]
            csv_file_name = output + '\\' + 'group_summary_log.csv'

            with open(csv_file_name, 'a') as myfile:
                writer = csv.DictWriter(myfile,
                                        fieldnames=['Time-Stamp', 'Time-Window', 'Group', 'LFP-Count', 'Tot-Channel',
                                                    'Act-Channel', 'LFP-Count-perCH', 'LFP-Count-per-Time',
                                                    'time-first-event', 'SZ-Channels', 'SZ-start', 'SZ-max-duration','SZ-mean-duration',
                                                    'SZ-distance', 'SZ-rate'])
                writer.writerow(groups)
                myfile.close()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=column_list, y=row_list, marker={'color': 'grey', 'showscale': False}, mode='markers', name='All Active Channels'))
            fig2.add_trace(go.Scatter(x=group_column, y=group_row, marker={'color': 'blue', 'showscale': False}, mode='markers', name='Group1 Channels'))
            fig2.add_trace(go.Scatter(x=rec_column, y=rec_row, marker={'color': 'red', 'showscale': False}, mode='markers', name='Channels Recruited'))
            fig2.add_trace(go.Scatter(x=start_column, y=start_row, marker={'color': 'green', 'showscale': False}, mode='markers', name='Point of Initiation'))
            fig2.update_xaxes(showline=True, linewidth=1, linecolor='black', range=[0, 65], mirror=True)
            fig2.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, range=[0, 65], autorange="reversed")
            fig2.update_layout(template="plotly_white", clickmode='event+select', width=800, height=800, legend=dict(orientation="h"))
            return fig2, [table_dict, ]
    else:
        return fig0, []


if __name__ == '__main__':
    app.run_server(debug=True)
