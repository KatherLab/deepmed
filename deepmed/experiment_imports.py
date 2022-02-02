import logging
import coloredlogs

coloredlogs.install(
    fmt='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s', level=logging.DEBUG)

from fastai.vision.all import *
from packaging.specifiers import SpecifierSet

from pathlib import Path
import pandas as pd
import deepmed
from deepmed import *
from deepmed import get, multi_input, evaluators, on_features, mil
from deepmed.evaluators import *
from deepmed.get import cohort