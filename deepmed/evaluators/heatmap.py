from typing import Optional, Iterable, Union, Tuple
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from scipy import interpolate
import regex as re
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def heatmap(
        target_label: str, preds_df: pd.DataFrame, path: Path,
        colors=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]]),
        wsi_paths: Optional[Iterable[Union[Path, str]]] = None,
        wsi_suffixes: Iterable[str] = ['.svs', '.ndpi'],
        superimpose: bool = False, format: str = '.svg') -> None:

    # openslide is non-trivial to install, so let's make it optional by lazily loading it
    from openslide import OpenSlide

    logger = logging.getLogger(str(path))
    outdir = path/'heatmaps'

    classes = sorted(preds_df[target_label].unique())
    score_labels = [f'{target_label}_{class_}' for class_ in classes]
    legend_elements = [
        Patch(facecolor=color, label=class_) for class_, color in zip(classes, colors)]

    for slide_name, tiles in preds_df.groupby('FILENAME'):
        true_label = tiles.iloc[0][target_label]
        try:
            plt.figure(dpi=600)
            slide_path = Path(tiles.tile_path.iloc[0]).parent
            map_coords = np.array([
                _get_coords(tile_path.name) for tile_path in slide_path.glob('*.jpg')])

            stride = _get_stride(map_coords)
            scaled_map_coords = map_coords // stride

            mask = np.zeros(scaled_map_coords.max(0) + 1)
            for coord in scaled_map_coords:
                mask[coord[0], coord[1]] = 1

            points = tiles.tile_path.map(lambda x: _get_coords(Path(x).name))
            points = np.array(list(points))

            points = points // stride

            values = tiles[score_labels].to_numpy()

            assert points.shape[1] == 2, "expected points to have shape (_, 2)"
            assert points.shape[0] == values.shape[0], \
                   "expected points and values to have the same number of elements"
            # grid which will form the basis for our output image
            grid_x, grid_y = np.mgrid[0:scaled_map_coords[:,0].max()+1,
                                      0:scaled_map_coords[:,1].max()+1]

            # interpolate heatmap over grid
            activations = interpolate.griddata(points, values, (grid_x, grid_y))
            activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

            if not wsi_paths:
                heatmap = _visualize_activation_map(
                    activations.transpose(1, 0, 2), colors[:activations.shape[-1]])
                heatmap = heatmap.resize(np.multiply(heatmap.size, 8), resample=Image.NEAREST)
                plt.imshow(heatmap)
                plt.axis('off')
                legend = plt.legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
            else:
                # find a wsi file with the slide
                fn = next(filter(Path.exists,
                                 ((Path(wsi_path)/slide_name).with_suffix(suffix)
                                  for wsi_path in wsi_paths
                                  for suffix in wsi_suffixes)),
                                 None)

                if fn is None: continue
                slide = OpenSlide(str(fn))

                # get the first level smaller than max_size
                level = next((i for i, dims in enumerate(slide.level_dimensions)
                              if max(dims) <= 2400*2),
                             slide.level_count-1)
                thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
                covered_area_size = (
                        (map_coords.max(0)+stride) /
                        slide.level_downsamples[level]).astype(int)
                heatmap = _visualize_activation_map(
                    activations.transpose(1, 0, 2),
                    colors=colors[:activations.shape[-1]],
                    alpha=.5 if superimpose else 1)

                scaled_heatmap = Image.new('RGBA', thumb.size)
                scaled_heatmap.paste(
                    heatmap.resize(covered_area_size, resample=Image.NEAREST))

                if superimpose:
                    thumb.alpha_composite(
                        scaled_heatmap)
                    plt.imshow(thumb)
                    plt.axis('off')
                    legend = plt.legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(12,6), dpi=300)
                    axs[0].imshow(thumb)
                    axs[0].axis('off')
                    axs[1].imshow(scaled_heatmap)
                    axs[1].axis('off')
                    legend = axs[1].legend(
                        title=target_label, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

            (outdir/true_label).mkdir(exist_ok=True, parents=True)
            fig.savefig((outdir/true_label/slide_name).with_suffix(format), bbox_extra_artists=[legend], bbox_inches='tight')
            plt.close('all')
        except Exception as exp:
            logger.exception(exp)


def _get_coords(filename: str) -> Optional[Tuple[int, int]]:
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', filename):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return (coords[0], coords[1]) # weird return format so mypy doesn't complain
    else: return None


def _get_stride(coordinates: np.array) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)
    return stride


def _visualize_activation_map(activations: np.ndarray, colors: np.ndarray, alpha: float = 1.) -> Image:
    """Transforms an activation map into an RGBA image.
    Args:
        activations: An (h, w, D) array of activations.
        colors: A (D, 3) array mapping each of the target classes to a color.
    Returns:
        An interpolated activation map image. Regions which the algorithm assumes to be background
        will be transparent.
    """
    assert colors.shape[1] == 3, "expected color map to have three color channels"
    assert colors.shape[0] == activations.shape[2], "one color map entry per class required"

    # transform activation map into RGB map
    rgbmap = activations.dot(colors)

    # create RGBA image with non-zero activations being the foreground
    mask = activations.any(axis=2)
    im_data = (np.concatenate([rgbmap, np.expand_dims(mask * alpha, -1)], axis=2) * 255.5).astype(np.uint8)

    return Image.fromarray(im_data)