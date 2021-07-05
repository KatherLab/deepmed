import logging
import coloredlogs

#FIXME this is a rather hacky way to set the global stderr handler log level
#logging.getLogger().handlers[0].setLevel(logging.DEBUG)
coloredlogs.install(
    fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s', level=logging.DEBUG)

from pathlib import Path
from deepest_histology import get
from deepest_histology.train import train
from deepest_histology.deploy import deploy
from deepest_histology.get import Cohort
from deepest_histology.experiment import do_experiment
from deepest_histology.metrics import *
from functools import partial
