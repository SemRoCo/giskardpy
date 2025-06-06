import os
import threading

def preload_matplotlib():
    # preload costly imports that are not used immediately
    import matplotlib.pyplot
    from scipy import sparse
    import pandas
    from pkg_resources import resource_filename

# Start preloading in the background
if 'GITHUB_WORKFLOW' not in os.environ:
    threading.Thread(target=preload_matplotlib, daemon=True).start()