import numpy as np
import h5py
import json
import os
import time
import scipy
import scipy.signal
import sys
import colorama

from colorama import Fore,Style,Back
colorama.init()

YELLOW = "\x1b[1;33;40m"
RED = "\x1b[1;31;40m"
BLUE = "\x1b[1;34; 40m"
print(f"\n{Fore.LIGHTBLUE_EX}Enter the Directory for emptyRaw.brw file: ", end='')
extDirectory = input()

class readBrw:
    def __init__(self, path):
        self.path = path
        self.data = []
        self.SamplingRate = []
        self.Chs = []
        self.NRecFrames = []
        self.MaxVolt = []
        self.MinVolt = []
        self.BitDepth = []
        self.raw = []
        self.fromQLevelToUVolt = []

    def load(self):
        self.data = h5py.File(self.path, 'r')
        self.SamplingRate = self.data['3BRecInfo/3BRecVars/SamplingRate'][()][0]
        self.Chs = self.data['3BRecInfo/3BMeaStreams/Raw/Chs'][:]
        self.NRecFrames = self.data['3BRecInfo/3BRecVars/NRecFrames'][0]
        self.MaxVolt = self.data['3BRecInfo/3BRecVars/MaxVolt'][0]  # in uVolt
        self.MinVolt = self.data['3BRecInfo/3BRecVars/MinVolt'][0]  # in uVolt
        self.BitDepth = self.data['3BRecInfo/3BRecVars/BitDepth'][0]
        self.raw = self.data['3BData/Raw'][:]

    def close(self):
        self.data.close()

    def conversionFactor(self):
        QLevel = np.power(2, self.BitDepth)
        return (self.MaxVolt - self.MinVolt) / QLevel

    def convertQlevel(self, dataQLevel):
        cF = self.conversionFactor()
        x = (dataQLevel - (np.power(2, self.BitDepth)) / 2) * cF
        return x

    def readChannel(self, ch):
        chSelected = (self.Chs['Row'] == ch[0]) * (self.Chs['Col'] == ch[1])
        refCh = np.where(chSelected == True)[0][0]
        numChs = len(self.Chs)
        idx = np.arange(refCh, self.NRecFrames * numChs, numChs)
        dataQLevel = self.raw[idx]
        return self.convertQlevel(dataQLevel)

    def recordingLengh(self):
        return self.NRecFrames / self.SamplingRate


class writeBrw:
    def __init__(self, path, name):
        self.path = path
        self.fileName = name

        self.brw = h5py.File(os.path.join(extDirectory, 'emptyRaw.brw'), 'r')

        self.signalInversion = self.brw['3BRecInfo/3BRecVars/SignalInversion']
        self.maxVolt = self.brw['3BRecInfo/3BRecVars/MaxVolt'][0]
        self.minVolt = self.brw['3BRecInfo/3BRecVars/MinVolt'][0]
        self.QLevel = np.power(2, self.brw['3BRecInfo/3BRecVars/BitDepth'][0])
        self.fromQLevelToUVolt = (self.maxVolt - self.minVolt) / self.QLevel

    def createNewBrw(self):
        newName = os.path.join(self.path, self.fileName + '.brw')
        new = h5py.File(newName, 'w')

        new.attrs.__setitem__('Description', self.brw.attrs['Description'])
        new.attrs.__setitem__('GUID', self.brw.attrs['GUID'])
        new.attrs.__setitem__('Version', self.brw.attrs['Version'])

        new.copy(self.brw['3BRecInfo'], dest=new)
        new.copy(self.brw['3BData'], dest=new)
        new.copy(self.brw['3BUserInfo'], dest=new)

        del new['/3BData/Raw']
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
    ADCCountsToMV = parameters['signalInversion'] * parameters['fromQLevelToUVolt']
    MVOffset = parameters['signalInversion'] * parameters['minUVolt']
    return ADCCountsToMV, MVOffset


def column_id(ch):
    with open('C:/Users/amahadevan/Documents/R-DriveData/channel_list.json') as file:
        full_ch_list = json.load(file)
    label = full_ch_list[str(ch)]
    return label


def downsample_channel(data,freq_ratio):
    s = time.time()
    re_sampleRatio = int(data.shape[0]/freq_ratio)
    data_downsampled = scipy.signal.resample(data, re_sampleRatio)
    return data_downsampled


if __name__ == '__main__':

    print(f"\n{YELLOW}Enter FOLDER PATH for Downsampling Files (Example - C:\\User): ", end='')
    folder = input()
    print(f"\n{Fore.RED}Enter New Sampling Frequency: ", end='')
    sampling_freq_input = input()
    count = 0
    for filename in os.listdir(folder):

        file = folder+'\\'+filename
        Input_file_path = file
        print()
        print(f"{Fore.GREEN}")
        print(count, 'Down Sampling File: ', Input_file_path)
        b = time.time()
        input_file_name = Input_file_path.split('\\')[-1]
        input_file_path = '\\'.join(Input_file_path.split('\\')[0:-1])
        
        new_sampling_frequency = int(sampling_freq_input)
        output_file_name = input_file_name.split('.')[0]+"_resample_"+str(new_sampling_frequency)
        output_file_name_brw = input_file_name.split('.')[0]+"_resample_"+str(new_sampling_frequency)+".brw"
        output_path = input_file_path+'\\'+output_file_name_brw
        #print(Input_file_path,output_path)

        h5 = h5py.File(Input_file_path,'r')
        parameters = parameter(h5)
        tot_frames = parameters['nRecFrames']
        tot_channels = len(parameters['recElectrodeList'])
        sampling = parameters['samplingRate']
        chList = parameters['recElectrodeList']
        freq_ratio = sampling / new_sampling_frequency


        dset = writeBrw(input_file_path, output_file_name)
        dset.createNewBrw()

        fs = new_sampling_frequency  # desired sampling frequency
        block_size = 100000
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
        
        for cnk in chunks:

            s = time.time()
            end = cnk * tot_channels

            data = np.array(h5['/3BData/Raw'][start:end])
            data = data.reshape(100000, tot_channels)
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
                fileName = os.path.join(input_file_path+"\\", output_file_name)
                dset.appendBrw(output_path, nrecFrame, res[ind, :])
            sys.stdout.write('\r')

            sys.stdout.write('Time Elapsed: ' + str(time.time()-b) + ' seconds  ')
            sys.stdout.flush()
            start = end
        print("\n Down Sampled Output File Location: ", output_path)
        count+=1
