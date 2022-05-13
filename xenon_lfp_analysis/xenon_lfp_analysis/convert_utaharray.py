import numpy as np
import h5py
import json
import os
import time
import scipy
import scipy.signal
import sys
from tqdm import tqdm
import click
from collections import namedtuple
from datetime    import datetime
from math        import ceil
from struct      import calcsize, pack, unpack, unpack_from
from qtpy.QtWidgets import QFileDialog, QApplication
from os             import getcwd, path
from os          import path as ospath


'''
#### Elements of this code are from: brpylib.py and brMiscFxns.py #### 

The functions required to load and extract the Utah Array compressed recording files are imported from the code *.ns5, .nev libraries. 

'''




# Define global variables to remove magic numbers
# <editor-fold desc="Globals">

# Define a named tuple that has information about header/packet fields
FieldDef = namedtuple('FieldDef', ['name', 'formatStr', 'formatFnc'])

WARNING_SLEEP_TIME      = 5
DATA_PAGING_SIZE        = 1024**3
DATA_FILE_SIZE_MIN      = 1024**2 * 10
STRING_TERMINUS         = '\x00'
UNDEFINED               = 0
ELEC_ID_DEF             = 'all'
START_TIME_DEF          = 0
DATA_TIME_DEF           = 'all'
DOWNSAMPLE_DEF          = 1
START_OFFSET_MIN        = 0
STOP_OFFSET_MIN         = 0

UV_PER_BIT_21             = 0.25
WAVEFORM_SAMPLES_21       = 48
NSX_BASIC_HEADER_BYTES_22 = 314
NSX_EXT_HEADER_BYTES_22   = 66
DATA_BYTE_SIZE            = 2
TIMESTAMP_NULL_21         = 0

NO_FILTER               = 0
BUTTER_FILTER           = 1
SERIAL_MODE             = 0

RB2D_MARKER             = 1
RB2D_BLOB               = 2
RB3D_MARKER             = 3
BOUNDARY_2D             = 4
MARKER_SIZE             = 5

DIGITAL_PACKET_ID       = 0
NEURAL_PACKET_ID_MIN    = 1
NEURAL_PACKET_ID_MAX    = 16384
COMMENT_PACKET_ID       = 65535
VIDEO_SYNC_PACKET_ID    = 65534
TRACKING_PACKET_ID      = 65533
BUTTON_PACKET_ID        = 65532
CONFIGURATION_PACKET_ID = 65531

PARALLEL_REASON         = 1
PERIODIC_REASON         = 64
SERIAL_REASON           = 129
LOWER_BYTE_MASK         = 255
FIRST_BIT_MASK          = 1
SECOND_BIT_MASK         = 2

CLASSIFIER_MIN          = 1
CLASSIFIER_MAX          = 16
CLASSIFIER_NOISE        = 255

CHARSET_ANSI            = 0
CHARSET_UTF             = 1
CHARSET_ROI             = 255

COMM_RGBA               = 0
COMM_TIME               = 1

BUTTON_PRESS            = 1
BUTTON_RESET            = 2

CHG_NORMAL              = 0
CHG_CRITICAL            = 1

ENTER_EVENT             = 1
EXIT_EVENT              = 2



def openfilecheck(open_mode, file_name='', file_ext='', file_type=''):
    """
    :param open_mode: {str} method to open the file (e.g., 'rb' for binary read only)
    :param file_name: [optional] {str} full path of file to open
    :param file_ext:  [optional] {str} file extension (e.g., '.nev')
    :param file_type: [optional] {str} file type for use when browsing for file (e.g., 'Blackrock NEV Files')
    :return: {file} opened file
    """

    while True:
        if not file_name:  # no file name passed

            # Ask user to specify a file path or browse
            file_name = input("Enter complete " + file_ext + " file path or hit enter to browse: ")

            if not file_name:
                if 'app' not in locals(): app = QApplication([])
                if not file_ext: file_type = 'All Files'
                file_name = QFileDialog.getOpenFileName(QFileDialog(), "Select File", getcwd(),
                                                        file_type + " (*" + file_ext + ")")

        # Ensure file exists (really needed for users type entering)
        if path.isfile(file_name):
            # Ensure given file matches file_ext
            if file_ext:
                _, fext = path.splitext(file_name)

                # check for * in extension
                if file_ext[-1] == '*': test_extension = file_ext[:-1]
                else:                   test_extension = file_ext

                if fext[0:len(test_extension)] != test_extension:
                    file_name = ''
                    print("\n*** File given is not a " + file_ext + " file, try again ***\n")
                    continue
            break
        else:
            file_name = ''
            print("\n*** File given does exist, try again ***\n")

    print('\n' + file_name.split('/')[-1] + ' opened')
    return open(file_name, open_mode)

def format_stripstring(header_list):
    string = bytes.decode(next(header_list), 'latin-1')
    return string.split(STRING_TERMINUS, 1)[0]

def format_filespec(header_list):
    return str(next(header_list)) + '.' + str(next(header_list))  # eg 2.3


def format_timeorigin(header_list):
    year        = next(header_list)
    month       = next(header_list)
    _           = next(header_list)
    day         = next(header_list)
    hour        = next(header_list)
    minute      = next(header_list)
    second      = next(header_list)
    millisecond = next(header_list)
    return datetime(year, month, day, hour, minute, second, millisecond * 1000)


def format_stripstring(header_list):
    string = bytes.decode(next(header_list), 'latin-1')
    return string.split(STRING_TERMINUS, 1)[0]


def format_none(header_list):
    return next(header_list)


def format_freq(header_list):
    return str(float(next(header_list)) / 1000) + ' Hz'


def format_filter(header_list):
    filter_type = next(header_list)
    if filter_type == NO_FILTER:        return "none"
    elif filter_type == BUTTER_FILTER:  return "butterworth"


def format_charstring(header_list):
    return int(next(header_list))


def format_digconfig(header_list):
    config = next(header_list) & FIRST_BIT_MASK
    if config: return 'active'
    else:      return 'ignored'


def format_anaconfig(header_list):
    config = next(header_list)
    if config & FIRST_BIT_MASK:  return 'low_to_high'
    if config & SECOND_BIT_MASK: return 'high_to_low'
    else:                        return 'none'


def format_digmode(header_list):
    dig_mode = next(header_list)
    if dig_mode == SERIAL_MODE: return 'serial'
    else:                       return 'parallel'


def format_trackobjtype(header_list):
    trackobj_type = next(header_list)
    if   trackobj_type == UNDEFINED:    return 'undefined'
    elif trackobj_type == RB2D_MARKER:  return '2D RB markers'
    elif trackobj_type == RB2D_BLOB:    return '2D RB blob'
    elif trackobj_type == RB3D_MARKER:  return '3D RB markers'
    elif trackobj_type == BOUNDARY_2D:  return '2D boundary'
    elif trackobj_type == MARKER_SIZE:  return 'marker size'
    else:                               return 'error'

def getdigfactor(ext_headers, idx):
    max_analog  = ext_headers[idx]['MaxAnalogValue']
    min_analog  = ext_headers[idx]['MinAnalogValue']
    max_digital = ext_headers[idx]['MaxDigitalValue']
    min_digital = ext_headers[idx]['MinDigitalValue']
    return float(max_analog - min_analog) / float(max_digital - min_digital)

nsx_header_dict = {
    'basic_21': [FieldDef('Label',              '16s', format_stripstring),   # 16 bytes  - 16 char array
                 FieldDef('Period',             'I',   format_none),          # 4 bytes   - uint32
                 FieldDef('ChannelCount',       'I',   format_none)],         # 4 bytes   - uint32

    'basic': [FieldDef('FileSpec',              '2B',   format_filespec),     # 2 bytes   - 2 unsigned char
              FieldDef('BytesInHeader',         'I',    format_none),         # 4 bytes   - uint32
              FieldDef('Label',                 '16s',  format_stripstring),  # 16 bytes  - 16 char array
              FieldDef('Comment',               '256s', format_stripstring),  # 256 bytes - 256 char array
              FieldDef('Period',                'I',    format_none),         # 4 bytes   - uint32
              FieldDef('TimeStampResolution',   'I',    format_none),         # 4 bytes   - uint32
              FieldDef('TimeOrigin',            '8H',   format_timeorigin),   # 16 bytes  - 8 uint16
              FieldDef('ChannelCount',          'I',    format_none)],        # 4 bytes   - uint32

    'extended': [FieldDef('Type',               '2s',   format_stripstring),  # 2 bytes   - 2 char array
                 FieldDef('ElectrodeID',        'H',    format_none),         # 2 bytes   - uint16
                 FieldDef('ElectrodeLabel',     '16s',  format_stripstring),  # 16 bytes  - 16 char array
                 FieldDef('PhysicalConnector',  'B',    format_none),         # 1 byte    - uint8
                 FieldDef('ConnectorPin',       'B',    format_none),         # 1 byte    - uint8
                 FieldDef('MinDigitalValue',    'h',    format_none),         # 2 bytes   - int16
                 FieldDef('MaxDigitalValue',    'h',    format_none),         # 2 bytes   - int16
                 FieldDef('MinAnalogValue',     'h',    format_none),         # 2 bytes   - int16
                 FieldDef('MaxAnalogValue',     'h',    format_none),         # 2 bytes   - int16
                 FieldDef('Units',              '16s',  format_stripstring),  # 16 bytes  - 16 char array
                 FieldDef('HighFreqCorner',     'I',    format_freq),         # 4 bytes   - uint32
                 FieldDef('HighFreqOrder',      'I',    format_none),         # 4 bytes   - uint32
                 FieldDef('HighFreqType',       'H',    format_filter),       # 2 bytes   - uint16
                 FieldDef('LowFreqCorner',      'I',    format_freq),         # 4 bytes   - uint32
                 FieldDef('LowFreqOrder',       'I',    format_none),         # 4 bytes   - uint32
                 FieldDef('LowFreqType',        'H',    format_filter)],      # 2 bytes   - uint16

    'data': [FieldDef('Header',                 'B',    format_none),         # 1 byte    - uint8
             FieldDef('Timestamp',              'I',    format_none),         # 4 bytes   - uint32
             FieldDef('NumDataPoints',          'I',    format_none)]         # 4 bytes   - uint32]
}



# <editor-fold desc="Header processing functions">

def processheaders(curr_file, packet_fields):
    """
    :param curr_file:      {file} the current BR datafile to be processed
    :param packet_fields : {named tuple} the specific binary fields for the given header
    :return:               a fully unpacked and formatted tuple set of header information

    Read a packet from a binary data file and return a list of fields
    The amount and format of data read will be specified by the
    packet_fields container
    """

    # This is a lot in one line.  First I pull out all the format strings from
    # the basic_header_fields named tuple, then concatenate them into a string
    # with '<' at the front (for little endian format)
    packet_format_str = '<' + ''.join([fmt for name, fmt, fun in packet_fields])

    # Calculate how many bytes to read based on the format strings of the header fields
    bytes_in_packet = calcsize(packet_format_str)
    packet_binary = curr_file.read(bytes_in_packet)

    # unpack the binary data from the header based on the format strings of each field.
    # This returns a list of data, but it's not always correctly formatted (eg, FileSpec
    # is read as ints 2 and 3 but I want it as '2.3'
    packet_unpacked = unpack(packet_format_str, packet_binary)

    # Create a iterator from the data list.  This allows a formatting function
    # to use more than one item from the list if needed, and the next formatting
    # function can pick up on the correct item in the list
    data_iter = iter(packet_unpacked)

    # create an empty dictionary from the name field of the packet_fields.
    # The loop below will fill in the values with formatted data by calling
    # each field's formatting function
    packet_formatted = dict.fromkeys([name for name, fmt, fun in packet_fields])
    for name, fmt, fun in packet_fields:
        packet_formatted[name] = fun(data_iter)

    return packet_formatted
# <editor-fold desc="Safety check functions">
def check_elecid(elec_ids):
    if type(elec_ids) is str and elec_ids != ELEC_ID_DEF:
        print("\n*** WARNING: Electrode IDs must be 'all', a single integer, or a list of integers.")
        print("      Setting elec_ids to 'all'")
        elec_ids = ELEC_ID_DEF
    if elec_ids != ELEC_ID_DEF and type(elec_ids) is not list:
        if type(elec_ids) == range: elec_ids = list(elec_ids)
        elif type(elec_ids) == int: elec_ids = [elec_ids]
    return elec_ids


def check_starttime(start_time_s):
    if not isinstance(start_time_s, (int, float)) or \
            (isinstance(start_time_s, (int, float)) and start_time_s < START_TIME_DEF):
        print("\n*** WARNING: Start time is not valid, setting start_time_s to 0")
        start_time_s = START_TIME_DEF
    return start_time_s


def check_datatime(data_time_s):
    if (type(data_time_s) is str and data_time_s != DATA_TIME_DEF) or \
            (isinstance(data_time_s, (int, float)) and data_time_s < 0):
        print("\n*** WARNING: Data time is not valid, setting data_time_s to 'all'")
        data_time_s = DATA_TIME_DEF
    return data_time_s


def check_downsample(downsample):
    if not isinstance(downsample, int) or downsample < DOWNSAMPLE_DEF:
        print("\n*** WARNING: Downsample must be an integer value greater than 0. "
              "      Setting downsample to 1 (no downsampling)")
        downsample = DOWNSAMPLE_DEF
    return downsample


def check_dataelecid(elec_ids, all_elec_ids):
    unique_elec_ids = set(elec_ids)
    all_elec_ids    = set(all_elec_ids)

    # if some electrodes asked for don't exist, reset list with those that do, or throw error and return
    if not unique_elec_ids.issubset(all_elec_ids):
        if not unique_elec_ids & all_elec_ids:
            print('\nNone of the elec_ids passed exist in the data, returning None')
            return None
        else:
            print("\n*** WARNING: Channels " + str(sorted(list(unique_elec_ids - all_elec_ids))) +
                  " do not exist in the data")
            unique_elec_ids = unique_elec_ids & all_elec_ids

    return sorted(list(unique_elec_ids))


def check_filesize(file_size):
    if file_size < DATA_FILE_SIZE_MIN:
        print('\n file_size must be larger than 10 Mb, setting file_size=10 Mb')
        return DATA_FILE_SIZE_MIN
    else:
        return int(file_size)
# </editor-fold>


class NsxFile:
    """
    attributes and methods for all BR continuous data files.  Initialization opens the file and extracts the
    basic header information.
    """

    def __init__(self, datafile=''):

        self.datafile         = datafile
        self.basic_header     = {}
        self.extended_headers = []

        # Run openfilecheck and open the file passed or allow user to browse to one
        self.datafile = openfilecheck('rb', file_name=self.datafile, file_ext='.ns*', file_type='Blackrock NSx Files')

        # Determine File ID to determine if File Spec 2.1
        self.basic_header['FileTypeID'] = bytes.decode(self.datafile.read(8), 'latin-1')

        # Extract basic and extended header information based on File Spec
        if self.basic_header['FileTypeID'] == 'NEURALSG':
            self.basic_header.update(processheaders(self.datafile, nsx_header_dict['basic_21']))
            self.basic_header['FileSpec']            = '2.1'
            self.basic_header['TimeStampResolution'] = 30000
            self.basic_header['BytesInHeader']       = 32 + 4 * self.basic_header['ChannelCount']
            shape = (1, self.basic_header['ChannelCount'])
            self.basic_header['ChannelID'] = \
                list(np.fromfile(file=self.datafile, dtype=np.uint32,
                                 count=self.basic_header['ChannelCount']).reshape(shape)[0])
        else:
            self.basic_header.update(processheaders(self.datafile, nsx_header_dict['basic']))
            for i in range(self.basic_header['ChannelCount']):
                self.extended_headers.append(processheaders(self.datafile, nsx_header_dict['extended']))
        
    def getdata(self, elec_ids='all', start_time_s=0, data_time_s='all', downsample=1, zeropad=False):
        """
        This function is used to return a set of data from the NSx datafile.

        :param elec_ids:      [optional] {list}  List of elec_ids to extract (e.g., [13])
        :param start_time_s:  [optional] {float} Starting time for data extraction (e.g., 1.0)
        :param data_time_s:   [optional] {float} Length of time of data to return (e.g., 30.0)
        :param downsample:    [optional] {int}   Downsampling factor (e.g., 2)
        :return: output:      {Dictionary} of:  data_headers: {list}        dictionaries of all data headers
                                                elec_ids:     {list}        elec_ids that were extracted (sorted)
                                                start_time_s: {float}       starting time for data extraction
                                                data_time_s:  {float}       length of time of data returned
                                                downsample:   {int}         data downsampling factor
                                                samp_per_s:   {float}       output data samples per second
                                                data:         {numpy array} continuous data in a 2D numpy array

        Parameters: elec_ids, start_time_s, data_time_s, and downsample are not mandatory.  Defaults will assume all
        electrodes and all data points starting at time(0) are to be read. Data is returned as a numpy 2d array
        with each row being the data set for each electrode (e.g. output['data'][0] for output['elec_ids'][0]).
        """

        # Safety checks
        start_time_s = check_starttime(start_time_s)
        data_time_s  = check_datatime(data_time_s)
        downsample   = check_downsample(downsample)
        elec_ids     = check_elecid(elec_ids)

        # initialize parameters
        output                          = dict()
        output['elec_ids']              = elec_ids
        output['start_time_s']          = float(start_time_s)
        output['data_time_s']           = data_time_s
        output['downsample']            = downsample
        output['data']                  = []
        output['data_headers']          = []
        output['ExtendedHeaderIndices'] = []

        datafile_samp_per_sec = self.basic_header['TimeStampResolution'] / self.basic_header['Period']
        data_pt_size          = self.basic_header['ChannelCount'] * DATA_BYTE_SIZE
        output['samp_per_s']  = float(datafile_samp_per_sec / downsample)

        elec_id_indices       = []
        front_end_idxs        = []
        analog_input_idxs     = []
        front_end_idx_cont    = True
        analog_input_idx_cont = True
        hit_start             = False
        hit_stop              = False
        d_ptr                 = {'BoH' : [], 'BoD' : []}
        # Move file position to start of datafile (if read before, may not be here anymore)
        self.datafile.seek(self.basic_header['BytesInHeader'], 0)

        # Based on FileSpec set other parameters
        if self.basic_header['FileSpec'] == '2.1':
            output['elec_ids'] = self.basic_header['ChannelID']
            output['data_headers'].append({})
            output['data_headers'][0]['Timestamp']     = TIMESTAMP_NULL_21
            output['data_headers'][0]['NumDataPoints'] = (ospath.getsize(self.datafile.name) - self.datafile.tell()) \
                                                         // (DATA_BYTE_SIZE * self.basic_header['ChannelCount'])
        else:
            output['elec_ids'] = [d['ElectrodeID'] for d in self.extended_headers]

        # Determine start and stop index for data
        if start_time_s == START_TIME_DEF: start_idx = START_OFFSET_MIN
        else:                              start_idx = int(round(start_time_s * datafile_samp_per_sec))
        if data_time_s == DATA_TIME_DEF:   stop_idx  = STOP_OFFSET_MIN
        else:                              stop_idx  = int(round((start_time_s + data_time_s) * datafile_samp_per_sec))

        # If a subset of electrodes is requested, error check, determine elec indices, and reduce headers
        if elec_ids != ELEC_ID_DEF:
            elec_ids = check_dataelecid(elec_ids, output['elec_ids'])
            if not elec_ids: return output
            else:
                elec_id_indices    = [output['elec_ids'].index(e) for e in elec_ids]
                output['elec_ids'] = elec_ids
        num_elecs = len(output['elec_ids'])

        # Determine extended header indices and idx for Front End vs. Analog Input channels
        if self.basic_header['FileSpec'] != '2.1':
            for i in range(num_elecs):
                idx = next(item for (item, d) in enumerate(self.extended_headers)
                           if d["ElectrodeID"] == output['elec_ids'][i])
                output['ExtendedHeaderIndices'].append(idx)

                if self.extended_headers[idx]['PhysicalConnector'] < 5: front_end_idxs.append(i)
                else:                                                   analog_input_idxs.append(i)

            # Determine if front_end_idxs and analog_idxs are contiguous (default = False)
            if any(np.diff(np.array(front_end_idxs)) != 1):     front_end_idx_cont    = False
            if any(np.diff(np.array(analog_input_idxs)) != 1):  analog_input_idx_cont = False

        # Pre-allocate output data based on data packet info (timestamp + num pts) and/or data_time_s
        # 1) Determine number of samples in all data packets to set possible number of output pts
        # 1a) For file spec > 2.1, get to last data packet quickly to determine total possible output length
        # 2) If possible output length is bigger than requested, set output based on requested
        if self.basic_header['FileSpec'] == '2.1':
            timestamp    = TIMESTAMP_NULL_21
            num_data_pts = output['data_headers'][0]['NumDataPoints']
        else :
            while self.datafile.tell() < ospath.getsize(self.datafile.name):
                d_ptr['BoH'].append(self.datafile.tell())
                self.datafile.seek(1, 1)
                if self.basic_header['FileSpec'] == '3.0' :
                    timestamp = unpack('<Q', self.datafile.read(8))[0]
                else :
                    timestamp = unpack('<I', self.datafile.read(4))[0]
                num_data_pts = unpack('<I', self.datafile.read(4))[0]
                d_ptr['BoD'].append(self.datafile.tell())
                output['data_headers'].append({'Timestamp' : timestamp, 'NumDataPoints' : num_data_pts})
                self.datafile.seek(num_data_pts * self.basic_header['ChannelCount'] * DATA_BYTE_SIZE, 1)

        # stop_idx_output = ceil(timestamp / self.basic_header['Period']) + num_data_pts
        # if data_time_s != DATA_TIME_DEF and stop_idx < stop_idx_output:  stop_idx_output = stop_idx
        # total_samps = int(ceil((stop_idx_output - start_idx) / downsample))
        
        for x in range(len(output['data_headers'])) :
            if (output['data_headers'][x]['NumDataPoints'] * self.basic_header['ChannelCount'] * DATA_BYTE_SIZE) > DATA_PAGING_SIZE:
                print("\nOutput data requested is larger than 1 GB, attempting to preallocate output now")

            # If data output is bigger than available, let user know this is too big and they must request at least one of:
            # subset of electrodes, subset of data, or use savensxsubset to smaller file sizes, otherwise, pre-allocate data
            self.datafile.seek(d_ptr['BoD'][x])
            data_length = output['data_headers'][x]['NumDataPoints']
            recorded_data_bytes = data_length * num_elecs * DATA_BYTE_SIZE;
            recorded_data = self.datafile.read(recorded_data_bytes)
            if zeropad == True :
                padsize = ceil(output['data_headers'][x]['Timestamp'] / self.basic_header['Period'])
                data_length += padsize
                zero_array = bytes(np.zeros((padsize*num_elecs,1), dtype=np.short))
            try:   np.zeros((data_length, num_elecs), dtype=np.short)
            except MemoryError as err:
                err.args += (" Output data size requested is larger than available memory. Use the parameters\n"
                             "              for getdata(), e.g., 'elec_ids', to request a subset of the data or use\n"
                             "              NsxFile.savesubsetnsx() to create subsets of the main nsx file\n", )
                raise
            if zeropad == True:
                recorded_data = zero_array + recorded_data
            output['data'].append(np.ndarray((data_length, num_elecs),
                                             '<h',
                                             recorded_data))

        return output



class writeBrw:
    def __init__(self,  outputFile):
        
        self.fileName = outputFile
        self.signalInversion = np.int64(1)
        self.maxVolt = np.int64(1000)
        self.minVolt = np.int64(0)
        self.QLevel = np.int64(np.power(2, 0))
        self.fromQLevelToUVolt = int((self.maxVolt - self.minVolt) / self.QLevel)


    def createNewBrw(self):
        newName = self.fileName
        new = h5py.File(newName, 'w')

        new.attrs.__setitem__('Description', 'Utah Array File')
        new.attrs.__setitem__('GUID', 'HDF5')
        new.attrs.__setitem__('Version', "UTAH-h5")

        new.create_dataset('/3BRecInfo/3BRecVars/SamplingRate', data=[np.int64(30000)])
        new.create_dataset('/3BRecInfo/3BRecVars/SignalInversion',data = [self.signalInversion])
        new.create_dataset('/3BRecInfo/3BRecVars/MaxVolt',data = [self.maxVolt])
        new.create_dataset('/3BRecInfo/3BRecVars/MinVolt',data = [self.minVolt])
        new.create_dataset('/3BRecInfo/3BRecVars/BitDepth',data = [np.int64(12)])
        self.newDataset = new


    def writeRaw(self, rawToWrite, typeFlatten='F'):

        newRaw = np.int16(rawToWrite.flatten(typeFlatten))

        self.newDataset.create_dataset('/3BData/Raw', data=newRaw, maxshape=(None,))

    def writeChs(self, chs):
        self.newDataset.create_dataset('/3BRecInfo/3BMeaStreams/Raw/Chs', data=chs)

    def witeFrames(self, frames):
        self.newDataset.create_dataset('/3BRecInfo/3BRecVars/NRecFrames', data=[np.int64(frames)])

    def writeSamplingFreq(self, fs):
        if self.newDataset['/3BRecInfo/3BRecVars/SamplingRate']:
            del self.newDataset['/3BRecInfo/3BRecVars/SamplingRate']

        self.newDataset.create_dataset('/3BRecInfo/3BRecVars/SamplingRate', data=[np.float64(fs)])

    def appendBrw(self, fName, frames, rawToAppend, typeFlatten='F'):
        brwAppend = h5py.File(fName, 'a')
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
        #self.brw.close()

def parameter(h5):
    parameters = {}
    #parameters['nRecFrames'] = h5['/3BRecInfo/3BRecVars/NRecFrames'][0]
    #parameters['samplingRate'] = h5['/3BRecInfo/3BRecVars/SamplingRate'][0]
    #arameters['recordingLength'] = parameters['nRecFrames'] / parameters['samplingRate']
    parameters['signalInversion'] = h5['/3BRecInfo/3BRecVars/SignalInversion'][0]  # depending on the acq version it can be 1 or -1
    parameters['maxUVolt'] = h5['/3BRecInfo/3BRecVars/MaxVolt'][0]  # in uVolt
    parameters['minUVolt'] = h5['/3BRecInfo/3BRecVars/MinVolt'][0]  # in uVolt
    parameters['bitDepth'] = h5['/3BRecInfo/3BRecVars/BitDepth'][0]  # number of used bit of the 2 byte coding
    parameters['qLevel'] = 2 ^ parameters['bitDepth']  # quantized levels corresponds to 2^num of bit to encode the signal
    parameters['fromQLevelToUVolt'] = (parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel']
    #parameters['recElectrodeList'] = list(h5['/3BRecInfo/3BMeaStreams/Raw/Chs'])  # list of the recorded channels
    #parameters['numRecElectrodes'] = len(parameters['recElectrodeList'])
    return parameters


def Digital_to_Analog(parameters):
    ADCCountsToMV = parameters['signalInversion'] * parameters['fromQLevelToUVolt']
    MVOffset = parameters['signalInversion'] * parameters['minUVolt']
    return ADCCountsToMV, MVOffset

def convert_to_digital(data, parameters):
    
    ADCCountsToMV, MVOffset = Digital_to_Analog(parameters)
    data = (data - MVOffset)/ADCCountsToMV
    return data

def generate_chMap():
    count = 1
    mapcount = 0
    chMap = {}
    for i in range(100):
        skip = [0,9,90,99]
        row = (mapcount // 10) + 1
        column = mapcount % 10 + 1 
        if i not in skip:
            chMap[count] = (row,column)
            count = count +1
        mapcount = mapcount+1
    return chMap

def downsample_channel(data,freq_ratio):
    re_sampleRatio = int(data.shape[0]/freq_ratio)
    data_downsampled = scipy.signal.resample(data, re_sampleRatio)
    return data_downsampled


@click.command()
@click.option('--filename','-f',prompt=True, help="Enter the full path of UTAH Array *.ns5 file to convert:")
@click.option('--newsampling','-ns', prompt=True, help="Enter DownSampling frequency in Hz:")
def run(filename,newsampling):
    s = time.time()
    pathNSF = filename
    pathH5 = pathNSF.split('.ns')[0]+"_resample"+ str(newsampling)+"_export.h5"
    NsxFileObj = NsxFile(pathNSF)
    output = NsxFileObj.getdata()
    dset = writeBrw(pathH5)
    dset.createNewBrw()
    parameters = {}
    parameters['signalInversion'] = np.int64(1)
    parameters['maxUVolt'] = np.int64(1000)
    parameters['minUVolt'] = np.int64(0)
    parameters['qLevel'] = np.int64(np.power(2, 0))
    parameters['fromQLevelToUVolt'] = np.int64((parameters['maxUVolt'] - parameters['minUVolt']) / parameters['qLevel'])
    

    chs = generate_chMap()
    fs = output['samp_per_s']
    freq_ratio = fs / int(newsampling)
    nrecFrame = output['data'][0].shape[0]
    data = output['data'][0]
    #data = convert_to_digital(data, parameters)
    newChs = np.zeros(len(chs), dtype=[('Row', '<i2'), ('Col', '<i2')])
    idx = 0
    for ch in chs:
        newChs[idx] = (np.int16(chs[ch][0]), np.int16(chs[ch][1]))
        idx += 1
 
    ind = np.lexsort((newChs['Col'], newChs['Row']))
    newChs = newChs[ind]
    blockSize = 100000
    range_value = nrecFrame//blockSize+1
    start = 0
    for cnk in tqdm(range(range_value),desc="Downsampling & Export Progress"):
        
        end = cnk*blockSize+blockSize
        res = data[start:end,:]
        resDS = downsample_channel(res, freq_ratio)
        resDS = np.transpose(resDS)
        frame = resDS.shape[1]
        if cnk*blockSize < blockSize:
            dset.writeSamplingFreq(newsampling)
            dset.writeRaw(resDS, typeFlatten='F')
            dset.witeFrames(frame)
            dset.writeChs(newChs)
            dset.close()
        else:
            dset.appendBrw(pathH5, frame, resDS)
        
        start = end


if __name__ == '__main__':
    run()