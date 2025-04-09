import threading

def preload_matplotlib():
    # preload costly imports that are not used immediately
    import matplotlib.pyplot
    from scipy import sparse
    import pandas
    from networkx.algorithms.shortest_paths.generic import shortest_path
    from networkx.drawing.nx_pydot import read_dot
    from pkg_resources import resource_filename

# Start preloading in the background
threading.Thread(target=preload_matplotlib, daemon=True).start()