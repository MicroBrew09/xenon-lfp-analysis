Installation & Setup 
====================

Xenon LFP Analysis Platform is a Plotly Dash application running on Python. The instruction below are for installing Python for *Windows*. *Linux* and *Mac OS* by default have a **Python** installation, you can skip to Step-3.

Windows Option 1: Using Python Environment
------------------------------------------
1. Download `Python>3.7x <https://www.python.org/downloads/windows/>`_
    .. image:: _static/pictures/installation1.png
        :width: 600px
        :align: center
        :height: 400px
        :alt: alternate text

2. Run the downloaded installer file, "Do Not check Install for all users", may need administrative permissions for all users. This will create a local environment for **Python** under **Windows** user account currently logged-in.
    .. image:: _static/pictures/installation2.png
            :width: 600px
            :align: center
            :height: 400px
            :alt: alternate text

#. Install **pip**:
    a. Once the **Python** installation is complete, open the *Command Prompt: cmd* 
    b. Check the **pip** installation:
    :: 
    
    >python -m pip –-version 

    c. If **pip** installation is not found:
    :: 
    
    >python get-pip.py

4. Install the required Packages:
:: 

>python -m install 'requirements.txt'

        


