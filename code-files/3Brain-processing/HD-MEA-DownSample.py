import numpy as np
import h5py
import json
import os
import time
import scipy
import scipy.signal
import sys
import colorama
from tqdm import tqdm
import click

from colorama import Fore,Style,Back
colorama.init()

YELLOW = "\x1b[1;33;40m"
RED = "\x1b[1;31;40m"
BLUE = "\x1b[1;34; 40m"

class writeBrw:

    '''
    Class to create .brw HDF5 file in the 3Brain BrainWave4 File system, includes function write new files, 
    and append files along with measurement configurations, ADC conversion, Frames, Sampling.

    '''
    
    def __init__(self, inputFilePath, outputFile):
        print(inputFilePath)
        self.path = inputFilePath
        self.fileName = outputFile
        self.brw = h5py.File(self.path, 'r')

        self.signalInversion = self.brw['3BRecInfo/3BRecVars/SignalInversion']
        self.maxVolt = self.brw['3BRecInfo/3BRecVars/MaxVolt'][0]
        self.minVolt = self.brw['3BRecInfo/3BRecVars/MinVolt'][0]
        self.QLevel = np.power(2, self.brw['3BRecInfo/3BRecVars/BitDepth'][0])
        self.fromQLevelToUVolt = (self.maxVolt - self.minVolt) / self.QLevel

    def createNewBrw(self):
        newName = self.fileName
        new = h5py.File(newName, 'w')

        new.attrs.__setitem__('Description', self.brw.attrs['Description'])
        new.attrs.__setitem__('GUID', self.brw.attrs['GUID'])
        new.attrs.__setitem__('Version', self.brw.attrs['Version'])

        new.copy(self.brw['3BRecInfo'], dest=new)
        #new.copy(self.brw['3BData'], dest=new)
        new.copy(self.brw['3BUserInfo'], dest=new)

        #del new['/3BData/Raw']
        del new['/3BRecInfo/3BMeaStreams/Raw/Chs']
        del new['/3BRecInfo/3BRecVars/NRecFrames']
        del new['/3BRecInfo/3BRecVars/SamplingRate']

        self.newDataset = new

    def writeRaw(self, rawToWrite, typeFlatten='F'):

        rawToWrite = rawToWrite / self.fromQLevelToUVolt
        rawToWrite = (rawToWrite + (self.QLevel / 2)) * self.signalInversion

        if rawToWrite.ndim == 1:
            newRaw = rawToWrite
        else:
            newRaw = np.int16(rawToWrite.flatten(typeFlatten))

        if '/3BData/Raw' in self.newDataset:
            dset = self.newDataset['3BData/Raw']
            dset.resize((dset.shape[0] + newRaw.shape[0],))
            dset[-newRaw.shape[0]:] = newRaw

        else:
            self.newDataset.create_dataset('/3BData/Raw', data=newRaw, maxshape=(None,))

    def writeChs(self, chs):
        self.newDataset.create_dataset('/3BRecInfo/3BMeaStreams/Raw/Chs', data=chs)

    def witeFrames(self, frames):
        self.newDataset.create_dataset('/3BRecInfo/3BRecVars/NRecFrames', data=[np.int64(frames)])

    def writeSamplingFreq(self, fs):
        self.newDataset.create_dataset('/3BRecInfo/3BRecVars/SamplingRate', data=[np.float64(fs)])

    def appendBrw(self, fName, frames, rawToAppend, typeFlatten='F'):

        brwAppend = h5py.File(fName, 'a')

        signalInversion = brwAppend['3BRecInfo/3BRecVars/SignalInversion']
        maxVolt = brwAppend['3BRecInfo/3BRecVars/MaxVolt'][0]
        minVolt = brwAppend['3BRecInfo/3BRecVars/MinVolt'][0]
        QLevel = np.power(2, brwAppend['3BRecInfo/3BRecVars/BitDepth'][0])
        fromQLevelToUVolt = (maxVolt - minVolt) / QLevel

        newFrame = brwAppend['/3BRecInfo/3BRecVars/NRecFrames'][0] + frames
        del brwAppend['/3BRecInfo/3BRecVars/NRecFrames']
        brwAppend.create_dataset('/3BRecInfo/3BRecVars/NRecFrames', data=[np.int64(newFrame)])

        if rawToAppend.ndim != 1:
            rawToAppend = np.int16(rawToAppend.flatten(typeFlatten))

        dset = brwAppend['3BData/Raw']
        dset.resize((dset.shape[0] + rawToAppend.shape[0],))
        dset[-rawToAppend.shape[0]:] = rawToAppend

        brwAppend.close()

    def close(self):
        self.newDataset.close()
        self.brw.close()

def parameter(h5):
    '''
    Generates a dictionary to store recording properties from the hdf5 file.

    Args:
        hdf5 File object in read mode (hdf5 object): 
            Example: h5 = h5py.File(filepath, 'r')

    Returns:
        recording properties dict (parameters): A python dict of recording parameters including sampling rate, 
        total recording frames, list of rows and column numbers of channels, digital to analog voltage conversion parameters. 
    
    '''
    parameters = {}
    parameters['nRecFrames'] = h5['/3BRecInfo/3BRecVars/NRecFrames'][0]
    parameters['samplingRate'] = h5['/3BRecInfo/3BRecVars/SamplingRate'][0]
    parameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
    parameters['signalInversion'] = h5['/3BRecInfo/3BRecVars/SignalInversion'][
        0]  # depending on the acq version it can be 1 or -1
    parameters['maxUVolt'] = h5['/3BRecInfo/3BRecVars/MaxVolt'][0]  # in uVolt
    parameters['minUVolt'] = h5['/3BRecInfo/3BRecVars/MinVolt'][0]  # in uVolt
    parameters['bitDepth'] = h5['/3BRecInfo/3BRecVars/BitDepth'][0]  # number of used bit of the 2 byte coding
    parameters['qLevel'] = 2 ^ parameters[
        'bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
    parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
    parameters['recElectrodeList'] = list(h5['/3BRecInfo/3BMeaStreams/Raw/Chs'])  # list of the recorded channels
    parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])
    return parameters


def Digital_to_Analog(parameters):
    '''
    Calculate and return the digital to analog conversion factor. 

        Args:
            parameters (dict): A python dict of recording parameters including sampling rate, 
            total recording frames, list of rows and column numbers of channels, digital to analog voltage conversion parameters.
        Returns:
            ADCCountsToMV (float): Analog to Digital Conversion Count (digital value to mV)
            MVOffset (float): Voltage offset (mV)  
    '''
    ADCCountsToMV = parameters['signalInversion'] * parameters['fromQLevelToUVolt']
    MVOffset = parameters['signalInversion'] * parameters['minUVolt']
    return ADCCountsToMV, MVOffset


def downsample_channel(data,freq_ratio):
    '''
    Scipy signal.resample() method to downsample inpt signal.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

        Args:
            data (np.ndarray): nd array of shape (block_size, NumChannels) containing the raw meaurement data.
            freq_ratio (np.float32): Ratio of original sampling frequency to new sampling frequency
        Returns:
            data_downsampled (np.ndarray): downsampled data to the resample ratio.  
    '''
    s = time.time()
    re_sampleRatio = int(data.shape[0]/freq_ratio)
    data_downsampled = scipy.signal.resample(data, re_sampleRatio)
    return data_downsampled


@click.command()
@click.option('--filename','-f',prompt=True, help="Enter the full path of the file to downsample:")
@click.option('--newsampling','-ns', prompt=True, help="Enter new sampling frequency in Hz:")
@click.option('--blocksize','-bs', default = 100000, prompt=False, help="Enter new sampling frequency in Hz:")

def main(filename,newsampling,blocksize):

    InputFilePath = filename
    print('Down Sampling File: ', InputFilePath)
    b = time.time()
    input_file_name = InputFilePath.split('\\')[-1]
    input_file_path = '\\'.join(InputFilePath.split('\\')[0:-1])
    new_sampling_frequency = int(newsampling)
    output_file_name_brw = input_file_name.split('.')[0]+"_resample_"+str(new_sampling_frequency)+".brw"
    output_path = input_file_path+'\\'+output_file_name_brw

    h5 = h5py.File(InputFilePath,'r')
    parameters = parameter(h5)
    tot_frames = parameters['nRecFrames']
    tot_channels = len(parameters['recElectrodeList'])
    sampling = parameters['samplingRate']
    chList = parameters['recElectrodeList']
    freq_ratio = sampling / new_sampling_frequency


    dset = writeBrw(InputFilePath, output_path)
    dset.createNewBrw()

    fs = new_sampling_frequency  # desired sampling frequency
    block_size = blocksize
    chunks = np.arange(block_size, tot_frames, block_size)

    channels = chList
    newChs = np.zeros(len(channels), dtype=[('Row', '<i2'), ('Col', '<i2')])

    idx = 0
    for ch in channels:
        newChs[idx] = (np.int16(ch[0]), np.int16(ch[1]))
        idx += 1

    # sort the channels for BrainWave 4
    ind = np.lexsort((newChs['Col'], newChs['Row']))
    newChs = newChs[ind]

    start = 0
    #nrecFrame = 0
    print(f"{Fore.GREEN}")

    for cnk in tqdm(chunks,desc="Downsampling & Export Progress"):
 
        end = cnk * tot_channels
        data = np.array(h5['/3BData/Raw'][start:end])
        data = data.reshape(block_size, tot_channels)
        data_resample = downsample_channel(data, freq_ratio)
        resamp_frame =  data_resample.shape[0]

        nrecFrame = resamp_frame
        res = np.zeros((len(channels),resamp_frame))

        ch = 0
        for channel in range(res.shape[0]):
            # for each channel store th information in uV
            res[channel, :] = data_resample[:, ch]
            ch +=1

        if cnk <= block_size:
            dset.writeRaw(res[ind, :], typeFlatten='F')
            dset.writeSamplingFreq(fs)
            dset.witeFrames(nrecFrame)
            dset.writeChs(newChs)
            dset.close()
        else:
            dset.appendBrw(output_path, nrecFrame, res[ind, :])

        start = end
    
    s = time.time()
    print("\n Down Sampled Output File Location: ", output_path)
    print("Time to complete:", round((s-b),2), "seconds")

if __name__ == '__main__':
    main()

