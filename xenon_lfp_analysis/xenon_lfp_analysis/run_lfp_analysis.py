import webbrowser
from threading import Timer
from pathlib import Path

from xenon_lfp_analysis.app import app

TITLE = "Xenon LFP Analysis Platform"

""" Xenon LFP Analysis 
"""

def main():
    """ Run the app from an entry point 
    
    """

    print(TITLE)

    # set up the url and a threading Timer
    host = "localhost"
    port = 7080
    folder = "app-name"
    url = f"http://{host}:{port}"
    Timer(10, webbrowser.open_new(url))

    # get back the location of the assets folder
    assets_folder = Path(app.__file__).parent / "assets"
    app.app.assets_folder = assets_folder

    # run app
    app.app.run_server(
        host=host,
        port=port,
        debug=False
    )