import shutil
from typing import Optional
from pathlib import Path

import pandas as pd
from PIL import Image

import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from PIL import Image


def top_tiles(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        n_patients: int = 4, n_tiles: int = 4, patient_label: str = 'PATIENT',
        best_patients: bool = True, best_tiles: Optional[bool] = None,
        save_images: bool = False) -> None:
    """Generates a grid of the best scoring tiles for each class.

    The function outputs a `n_patients` × `n_tiles` grid of tiles, where each
    row contains the `n_tiles` highest scoring tiles for one of the `n_patients`
    best-classified patients.

    Args:
        best_patients:  Wether to select the best or worst n patients.
        best_tiles:  Whether to select the highest or lowest scoring tiles.  If
            set to ``None``, then the same as ``best_patients``.
        save_images:  Also save the tiles seperately.
    """
    # set `best_tiles` to `best_patients` if undefined
    best_tiles = best_tiles if best_tiles is not None else best_patients

    for class_ in preds_df[f'{target_label}_pred'].unique():
        # class_ == MSIH
        outdir = result_dir/_generate_tiles_fn(
                target_label, class_, best_patients, best_tiles, n_patients, n_tiles)
        outfile = Path(str(outdir) + '.svg')
        if outfile.exists() and (outdir.exists() or not save_images):
            continue
        if save_images:
            outdir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(n_tiles, n_patients), dpi=600)
        # get patients with the best overall ratings for the label
        class_instance_df = preds_df[preds_df[target_label] == class_]
        patient_scores = \
            class_instance_df.groupby(patient_label)[f'{target_label}_pred'].agg(lambda x: sum(x == class_) / len(x))

        patients = (patient_scores.nlargest(n_patients) if best_patients
                    else patient_scores.nsmallest(n_patients))

        top_tile_list = []
        for i, patient in enumerate(patients.keys()):
            # get the best tile for that patient
            patient_tiles = preds_df[preds_df[patient_label] == patient]

            tiles = (patient_tiles.nlargest(n=n_tiles, columns=f'{target_label}_{class_}')
                     if best_tiles
                     else patient_tiles.nsmallest(n=n_tiles, columns=f'{target_label}_{class_}'))
            top_tile_list.append(tiles)

            for j, tile in enumerate(tiles.tile_path):
                if save_images:
                    shutil.copy(tile, outdir/Path(tile).name)
                if not outfile.exists():
                    plt.subplot(n_patients, n_tiles, i*n_tiles + j+1)
                    plt.axis('off')
                    plt.imshow(Image.open(tile), cmap='gray')

        pd.concat(top_tile_list).to_csv(outfile.with_suffix('.csv'), index=False)

        if not outfile.exists():
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def _generate_tiles_fn(
        target_label: str, class_: str, best_patients: bool, best_tiles: bool,
        n_patients: int, n_tiles: int) -> str:
    patient_str = f'{"best" if best_patients else "worst"}-{n_patients}-patients'
    tile_str = f'{"best" if best_tiles else "worst"}-{n_tiles}-tiles'

    return f'{target_label}_{class_}_{patient_str}_{tile_str}'