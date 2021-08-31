import logging
import coloredlogs

coloredlogs.install(
    fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s', level=logging.INFO)

from fastai.vision.all import *
from packaging.specifiers import SpecifierSet

from pathlib import Path
from functools import partial
import pandas as pd
import deepmed
from deepmed import *
from deepmed import get, metrics, multi_input, evaluators
from deepmed.metrics import *
from deepmed.evaluators import *
from deepmed.get import cohort