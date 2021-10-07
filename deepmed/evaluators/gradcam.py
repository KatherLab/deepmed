import shutil
from typing import Optional
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt 

from fastai.vision.all import *
from fastai.torch_core import *
from deepmed.multi_input import MultiInputModel
from .top_tiles import _generate_tiles_fn

class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go):
        self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

def gradcam(
        target_label: str, preds_df: pd.DataFrame, result_dir: Path,
        n_patients: int = 4, n_tiles: int = 4, patient_label: str = 'PATIENT',
        best_patients: bool = True, best_tiles: Optional[bool] = None,
        save_images: bool = False) -> None:
    """Generates a grid of GRAD CAM images for the best scoring tiles for each class.

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
        outdir = result_dir/_generate_tiles_fn(
                target_label, class_, best_patients, best_tiles, n_patients, n_tiles)
        outfile = Path(str(outdir) + '_GradCAM.svg')
        
        if outfile.exists() and (outdir.exists() or not save_images):
            continue
        if save_images:
            outdir.mkdir(parents=True, exist_ok=True)

        tile_dict = dict()
        
        class_instance_df = preds_df[preds_df[target_label] == class_]
        patient_scores = \
            class_instance_df.groupby(patient_label)[f'{target_label}_pred'].agg(lambda x: sum(x == class_) / len(x))

        patients = (patient_scores.nlargest(n_patients) if best_patients
                    else patient_scores.nsmallest(n_patients))

        for i, patient in enumerate(patients.keys()):
            patient_tiles = preds_df[preds_df[patient_label] == patient]

            tiles = (patient_tiles.nlargest(n=n_tiles, columns=f'{target_label}_{class_}')
                     if best_tiles
                     else patient_tiles.nsmallest(n=n_tiles, columns=f'{target_label}_{class_}'))


            for j, (_, tile) in enumerate(tiles.iterrows()):
                #if hasattr(tile, "fold"):
                #    n_fold = tile.fold
                #    p_fold = result_dir/f"fold_{n_fold}"/"export.pkl"
                #else:
                p_fold = result_dir/"export.pkl"
    
                learn = load_learner(p_fold) ## p/target , cpu=False
                
                dls = learn.dls
                
                if hasattr(dls.vocab, 'o2i'):
                    cls_dec = dls.vocab.o2i[class_]
                else:
                    try: 
                        dict_vocab = {w:i for i,w in enumerate(dls.vocab)}
                        cls_dec = dict_vocab[str(class_)]
                    except TypeError: #not hashable, categorymap
                        dict_vocab = {w:i for i,w in enumerate(dls.vocab[-1])}
                        cls_dec = dict_vocab[str(class_)]

                x = first(dls.test_dl(tile.to_frame().transpose()))
                
                # TODO: referencing MultiInputModel
                
                if isinstance(learn.model, MultiInputModel):
                    feature_extractor = learn.model.cnn_feature_extractor[0]
                else:
                    feature_extractor = learn.model[0]
                    
                with HookBwd(feature_extractor) as hookg:
                    with Hook(feature_extractor) as hook:
                        output = learn.model.eval()(*x) #.cuda()
                        act = hook.stored
                    output[0, cls_dec].backward()
                    grad = hookg.stored
                
                w = grad[0].mean(dim=[1,2], keepdim=True) 
                cam_map = (w * act[0]).sum(0)
                
                x_dec = TensorImage(dls.train.decode(x)[0][0])
                _,ax = plt.subplots()
                x_dec.show(ctx=ax)
                
                ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,x[0].shape[2],x[0].shape[3],0), interpolation='bilinear', cmap='magma');
    
                plt.axis('off')
                
                index =  i*n_tiles + j+1
                tile_dict[index] = tile
                tile_name = Path(tile.tile_path).stem
                # if save_images == True: 
                gradcam_dir = result_dir / 'Grad-CAM_images'
                gradcam_dir.mkdir(exist_ok=True)
                out_pic = gradcam_dir/f"{tile_name}_{class_}_Grad-CAM.png" 
            
                tile_dict[index] = out_pic 
                plt.savefig(out_pic)
                plt.close()
        
        if not outfile.exists():
            plt.figure(figsize=(n_patients, n_tiles), dpi=600)
            for i, im in tile_dict.items():
                plt.subplot(n_patients, n_tiles, i)
                plt.axis('off')
                plt.imshow(Image.open(im)) # cannot read svg thus atm gradcams are saved as PNG imgs
            plt.savefig(outfile, bbox_inches='tight')
        
        plt.close()
