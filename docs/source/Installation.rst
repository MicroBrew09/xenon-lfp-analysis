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
The current example is illustrated to work with the 3Brain BioCAM-X recording using the BrainWave4 Software. \
The code is provided in the `github Repository: <https://github.com/MicroBrew09/xenon-lfp-analysis.git>`