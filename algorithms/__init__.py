""" Fetches all python modules inside the package. """
from os.path import basename, dirname, isfile, join
from glob import glob

modules = glob(join(dirname(__file__), "*.py"), recursive=False)
__all__ = [basename(m)[:-3] for m in modules if isfile(m) and basename(m)[:-3] != "__init__"]