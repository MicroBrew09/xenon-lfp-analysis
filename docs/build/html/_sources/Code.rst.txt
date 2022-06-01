Additional Pre-processing Tools
===============================

Xenon-LFP-Analysis GUI and the code files dicussed in the Tutorial sections, works with uncompressed-RAW recording,\
however at times the recording for large duration measurements may be in the \
BrainWave4 or BrainWave5 proprietary Wavelet Compressed format, this requires the BrainWave4 or BrainWave5 Decompression class and methods,\
in this section we provide additional support files to select and export a set of channels from the RAW or Wavelet compressed format to the uncompressed RAW hdf5 file,\
that can be analysed in the Xenon LFP Analysis GUI.\
The code files generally works for measurements collected using the BrainWave4 and BrainWave5 software version, both on the RAW and WaveletCompressed files, \
however will require Python version 3.7 and the BrainWave5 software installed to uncompress the proprietary Wavelet compression data. \
\

It is a two step process to select, and then extract the channels from the large compressed measurement file. \
In the first step, a light weight interactive application *Select, Downsample and Export: Channel Selection Toolbox* is used to upload the slice image overlay, \
and select channels that need to be exported, this generates a *xx_exportCh* hdf5 file. In the second step, \ 
the measurement file along with the generated *xx_exportCh* hdf5 files are placed in a folder and the provided Python script is run \
to extract and downsample the selected channels. When you have multiple measurements files, you can have several pairs of measurement files and the selection file (xx_exportCh) \
in the same folder to process them as a batch. 

Select channels to export
--------------------------

A. Download or clone the code-files from GitHub-Repository, \
run the below command using the full path of the Python code file *ExportToHDF5-ChannelSelection.py*. 
::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\3Brain-processing\\ExportToHDF5-ChannelSelection.py


Copy and paste http://127.0.0.1:9090/ in the browser (Firefox or Chrome or Edge).



1. Once the Dash application is running it should look as below:
    .. image:: _static/pictures/Capture1.PNG
        :width: 600px
        :align: center
        :height: 400px
        :alt: alternate text  

2. You will need two inputs, first the *full folder path and file name* and second an image file to over lay on the slice using the *upload image* icon, to upload the cropped slice image (if you don't have a slice image you still need to upload a dummy image to get to the next step):
    .. image:: _static/pictures/Capture2.PNG
        :width: 600px
        :align: center
        :height: 400px
        :alt: alternate text 

3. You can now use the *lasso* tool or *box tool* to select channels to export, you can hold down the *shift key* to select multiple regions or un-check selected channels.  
    .. image:: _static/pictures/Capture3.PNG
        :width: 600px
        :align: center
        :height: 400px
        :alt: alternate text 
4. Once you have selected the channels, you can set the downsampling frequency, or reduce the number of channels in the region using the options provided and click **Export Channels to *.brw File**.  
    .. image:: _static/pictures/Capture4.PNG
        :width: 600px
        :align: center
        :height: 400px
        :alt: alternate text 


B. Extract selected channels from BrainWave file recording
-----------------------------------------------------------
