Installation, Setup and Running Dash 
====================================

Xenon LFP Analysis Platform is a Plotly Dash application running on **Python**. The instruction below are for installing Python for *Windows*. *Linux* and *Mac OS* by default have a **Python** installation, you can skip to installing the **Python** libraries from the *requirements.txt* file.

Python Installation
-------------------

Option 1: Using **Python** for **Windows** Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download `Python>3.7x <https://www.python.org/downloads/windows/>`_
    .. image:: _static/pictures/installation1.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text

2. Run the downloaded installer file, **Do Not** check "Install launcher for all users (recommended)", may need administrative permissions for all users. This will create a local environment for **Python** under **Windows** user account currently logged-in.
    .. image:: _static/pictures/installation2.png
            :width: 300px
            :align: center
            :height: 200px
            :alt: alternate text

3. Install **pip**:

    a. Once the **Python** installation is complete, open the *Command Prompt*:
        .. image:: _static/pictures/installation7.png
            :width: 300px
            :align: center
            :height: 200px
            :alt: alternate text
            
    b. Check the **pip** installation:
    :: 

    >python -m pip –-version 

    c. If **pip** installation is not found:
    :: 

    >python get-pip.py

4. Install the required **Python** libraries:
:: 

>python -m install 'requirements.txt'

Option 2: Using Anaconda for **Windows** and **MacOS** Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download the appropriate `Anaconda-Version <https://www.anaconda.com/products/individual#windows>`_
    .. image:: _static/pictures/installation3.png
        :width: 600px
        :align: center
        :height: 200px
        :alt: alternate text

2. Run the installer file:
    .. image:: _static/pictures/installation4.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text

3. Select "Just Me (recommended)"" and continue to complete the installation:
    .. image:: _static/pictures/installation5.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text

4. Select and open "Anaconda Prompt" from the *Windows* 'Start' Menu:
    .. image:: _static/pictures/installation6.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text

5.	Install the required **Python** libraries:
::

>python -m pip install 'requirements.txt'


Export Channels For Analysis 
----------------------------

The full recording from the HD MEA platform can range from 80 GB to 250GB uncompressed and will not fit in the systems local memory (RAM) for analysis. \
The Dash application can only work within the capacity of the local memory, for this we export a subset of channels that are of interest and \
downsample the traces to less than 2048 Hz sampling frequency. This gives us a  frequency range of upto 1024 Hz (2048/2 Nyquist Frequency), also the maximum sampling frequency is not \
limited by the processing capability of the application, but limited to rendering inteactive scatter plots with a large number of data points in the browser. \
For example the application can work with two or three traces of sampled at 10000 Hz, or about 200 traces at 2048 Hz or 600 to 1000 traces at a sampling frequency of 300 Hz. 

HD-MEA Recording using the 3Brain BioCAM-X Measurement System:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This example below is illustrated for the BrainWave4 Software, other aquisition systems may have a different process. 

1. Open BrainWave4 Software application:
    .. image:: _static/pictures/installation8.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text  

2. Open the compressed or uncompressed **.brw** *HDF5* recording file in the application:
    .. image:: _static/pictures/installation9.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text 

3. Export a group of channel: **File-> Export-> Raw Data** or **Ctr + E**, and select the subset of channels of interest, give it a file name and **Export**.  
    .. image:: _static/pictures/installation10.png
        :width: 300px
        :align: center
        :height: 200px
        :alt: alternate text 
    

Downsample Exported Channels for Analysis
-----------------------------------------

BrainWave4 .brw HDF5 Files (3Brain - BioCAM-X Measurement):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current example is illustrated to work with the 3Brain BioCAM-X recording using the BrainWave4 Software. \
The code is provided in the `GitHub Repository: <https://github.com/MicroBrew09/xenon-lfp-analysis.git>`_

1.	Open Windows Command Prompt (cmd) if you are using Python base installation, or \
Open Anaconda Command Prompt (anaconda) if you are working with Anaconda.

    .. image:: _static/pictures/installation11.png
        :width: 800px
        :align: center
        :height: 300px
        :alt: alternate text 

2. Download or clone the code-files from GitHub-Repository, \
run the below command using the full path of the code file HD-MEA-DownSample.py. 
::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\HD-MEA-DownSample.py -f \\file-path\\Slice1_raw.brw :str -ns SamplingFrequency: int -bs blocksize: default 100000 

or 

::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\HD-MEA-DownSample.py

Running Dash and Xenon LFP Analysis Platform 
---------------------------------------------

BrainWave4 .brw HDF5 Files (3Brain - BioCAM-X Measurement):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The current example is illustrated to work with the 3Brain BioCAM-X recording using the BrainWave4 Software. \
The code is provided in the `GitHub Repository: <https://github.com/MicroBrew09/xenon-lfp-analysis.git>`_

While Dash applications can be deployed on a server and run remotely, it does not work well in this case, particularly on the .brw and large HDF5 files. \
The HDF5 files are not suitable for multiple parallel reads, or loading and transmitting data over a network, \ 
it is quite challenging to load and transmit large chunks of the HDF5 files back and forth between the remote server and local storage.
In this case we have found it inefficient, sometimes the file does not load, when the application is deployed on a remote server. \
For efficient analysis, the Dash application needs to run on the \
local machine and accessed through local host as below, it is also recommended that data files are present in the local hard-drive.

1.	Open Windows Command Prompt (cmd) if you are using Python base installation, or \
Open Anaconda Command Prompt (anaconda) if you are working with Anaconda.

    .. image:: _static/pictures/installation11.png
        :width: 800px
        :align: center
        :height: 300px
        :alt: alternate text 

2. Download or clone the code-files from GitHub-Repository, \
run the below command using the full path of the code file HD-MEA-DownSample.py. 
::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\Xenon-LFP-Analysis.py

    .. image:: _static/pictures/installation12.png
        :width: 500px
        :align: center
        :height: 100px
        :alt: alternate text 

3. Copy and paste http://127.0.0.1:8050/ in the browser (Firefox or Chrome).  

Repeat steps 1 to 3 if the program crashes or you want to restart analysis.\