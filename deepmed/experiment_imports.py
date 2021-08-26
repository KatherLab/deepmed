import logging
import coloredlogs

coloredlogs.install(
    fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s', level=logging.INFO)

from fastai.vision.all import *

from pathlib import Path
from functools import partial
import pandas as pd
from deepmed import *
from deepmed.metrics import *
from deepmed.get import cohort