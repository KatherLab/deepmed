from typing import Union, Optional
from pathlib import Path

PathLike = Union[str, Path]

class Cohort:
    def __init__(self,
                 root_dir: PathLike,
                 block_dir: PathLike,
                 clini_table: Optional[PathLike] = None,
                 slide_table: Optional[PathLike] = None) -> None:
        self.root_dir = Path(root_dir)
        self.block_dir = self.root_dir/block_dir

        # search for clini / slide tables if none are given
        if clini_table:
            self.clini_table = self.root_dir/clini_table
        else:
            for ext in ['.csv', '.xlsx']:
                if (path := (self.root_dir/f'{self.root_dir.stem}_CLINI').with_suffix(ext)).is_file():
                    self.clini_table = path
                    break
            else:
                raise ValueError(f'No clini table found for {root_dir}!')

        if slide_table:
            self.slide_table = self.root_dir/slide_table
        else:
            for ext in ['.csv', '.xlsx']:
                if (path := (self.root_dir/f'{self.root_dir.stem}_SLIDE').with_suffix(ext)).is_file():
                    self.slide_table = path
                    break
            else:
                raise ValueError(f'No slide table found for {root_dir}!')