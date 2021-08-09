import logging
import coloredlogs

coloredlogs.install(
    fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s', level=logging.INFO)

from pathlib import Path
from functools import partial
import pandas as pd
from deepest_histology import *
from deepest_histology.metrics import *
from deepest_histology.get import cohort