Installation, Setup and Running Xenon LFP Analysis GUI 
======================================================
Xenon LFP Analysis Platform is a Plotly Dash application running on **Python**. \

Quick Start
------------

1. Install **Python** and **pip** module if not already installed.


2. Install Xenon Lfp Analysis package

    :: 

        >pip install xenon_lfp_analysis

3. To start the gui:
    
    ::

        >run_lfp_analysis

4. This should open the gui in the browser, you can view the video tutorials to get started. 

Python Installation
-------------------
The instruction below are for installing Python for *Windows*. \ 
*Linux* and *Mac OS* by default have a **Python** installation, you can skip to installing the **Python** libraries from the *requirements.txt* file.

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

Download and Run Executable:
----------------------------
While Dash applications can be deployed on a server and run remotely, it does not work well in this case, particularly on the .brw and large HDF5 files. \
The HDF5 files are not suitable for multiple parallel reads, or loading and transmitting data over a network, \ 
it is quite challenging to load and transmit large chunks of the HDF5 files back and forth between the remote server and local storage.
In this case we have found it inefficient, sometimes the file does not load, when the application is deployed on a remote server. \
For efficient analysis, the Dash application needs to run on the \
local machine and accessed through local host using the *pip installation package*, or the executable file as below,\
it is also recommended that data files are present in the local hard-drive.

Windows
~~~~~~~~
a. Download the application `Xenon-LFP-Analysis-Windows <https://drive.google.com/file/d/17yPDSz-fjV8okBTVj0NMd-0fA4f1YWy1/view?usp=sharing>`_

b. Copy and Paste the url (http://127.0.0.1:7080) in any browser to run the application:

MacOS
~~~~~~

a. Download the application `Xenon-LFP-Analysis-MacOS <https://drive.google.com/file/d/1hjn7Xm4I3NwcZxlnSF4ORpJE2ovUWeIz/view?usp=sharing>`_

b. Open a new terminal window, cd into the downloaded file location.

c. Run the following command to change/confirm it is a Unix Executable.

    ::

    $ chmod u+x Xenon-LFP-Analysis

d. Now you can double click on the file to run the application. 

    If you get a security warning, click on the Apple logo -> System Preferences -> Security & Privacy. \
    At the bottom of the window, press 'Allow' to allow the file to run.

d. Copy and Paste the url (http://127.0.0.1:7080) in any browser to run the application


Linux (Ubuntu/Debian)
~~~~~~~~~~~~~~~~~~~~~

    a. Download the application `Xenon-LFP-Analysis-Ubuntu <https://drive.google.com/file/d/1kVrFbxkJt-2xlmnf64OJF8_vSbzs7_Cs/view?usp=sharing>`_

    b. Open a new terminal window, cd into the downloaded file location.
    
    c. Run the following command to change/confirm it is a Linux Executable.

    ::

    $ chmod u+x Xenon-LFP-Analysis-ubuntu

    d. Run the application

    ::

        $./Xenon-LFP-Analysis

    .. .. image:: _static/pictures/installation14.png
    ..     :width: 400px
    ..     :align: center
    ..     :height: 100px
    ..     :alt: alternate text

    d. Copy and Paste the url http://127.0.0.1:7080 in any browser. 


Export Channels For Analysis 
----------------------------

The full recording from the HD MEA platform can range from 80 GB to 250GB uncompressed and will not fit in the systems local memory (RAM) for analysis. \
The Dash application can only work within the capacity of the local memory, with the raw uncompressed data, for this we export a subset of channels that are of interest and \
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

2. If you have the the **xenon_lfp_analysis** package installed you can run the following to downsample *\*.brw file*

:: 

>downsample_brw

or  

2. Download or clone the code-files from GitHub-Repository, \
run the below command using the full path of the code file HD-MEA-DownSample.py. 
::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\HD-MEA-DownSample.py -f \\file-path\\Slice1_raw.brw :str -ns SamplingFrequency: int -bs blocksize: default 100000 

or 

::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\HD-MEA-DownSample.py


HD-MEA Recording using the UTAH Array Measurement System:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sample Utah-Array (BlackRock Microsystems) recording is provided in the `*.ns5 format <https://www.dropbox.com/s/5a9ufj09nzpehjm/Iqseq_id520_007.ns5?dl=0>`_ , \
a `code file <https://github.com/MicroBrew09/xenon-lfp-analysis.git>`_ is provided to extract the data \ 
and downsample to a lower frequency for analysis with the Xenon-LFP-Analysis platform. 

1. Open Windows Command Prompt (cmd) if you are using Python base installation, or \
Open Anaconda Command Prompt (anaconda) if you are working with Anaconda.

2. Download or clone the code-files from GitHub-repository, \
run the below command using the full path of the code file utah-array-conversion.py. \
You can either use the *-f* and *-ns* tags to provide the file name and new sampling frequency in Hz, or \
when promted enter the full file path for the *.ns5* file, \
and enter the desired sampling frequency. 
::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\utah-array-conversion.py -f \\file-path\\Slice1.ns5 :str -ns SamplingFrequency: int  

or 

::

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\utah-array-conversion.py

3. The `downsampled file <https://drive.google.com/file/d/1Ao1xW8prK4MasGJazjKWqtPbCJniIjBH/view?usp=sharing>`_ can be now be run on the Xenon-LFP-Analysis platform.\
The code may need a few modifications for larger recordings, \
the provided example is only a sample and may vary by use case, \
feel free to email or contact us if you run into issues. 

Running Dash and Xenon LFP Analysis Platform 
---------------------------------------------

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

>python C:\\Downloads\\xenon-lfp-analysis\\code-files\\Xenon-LFP-Analysis.py

    .. image:: _static/pictures/installation12.png
        :width: 500px
        :align: center
        :height: 100px
        :alt: alternate text 

3. Copy and paste http://127.0.0.1:8050/ in the browser (Firefox or Chrome).  

Repeat steps 1 to 3 if the program crashes or you want to restart analysis.\
