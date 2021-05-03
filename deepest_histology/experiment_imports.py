import logging

logging.basicConfig(level=logging.DEBUG)
#FIXME this is a rather hacky way to limit the global stderr handler
logging.getLogger().handlers[0].setLevel(logging.INFO)

from pathlib import Path
from deepest_histology import *
from deepest_histology.evaluate import *