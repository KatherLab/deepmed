from typing import Callable, Optional
import pandas as pd
from pathlib import Path

Evaluator = Callable[[Optional[str], pd.DataFrame, Path], Optional[pd.DataFrame]]