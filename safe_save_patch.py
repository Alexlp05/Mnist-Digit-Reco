"""
Patch pour résoudre les problèmes de PermissionError avec safe_save sur Windows
"""
import time
import pathlib
from typing import Dict
from tinygrad import Tensor
from tinygrad.nn.state import safe_save as tinygrad_safe_save

def safe_save_windows(tensors: Dict[str, Tensor], fn, max_retries: int = 5) -> None:
    """
    Wrapper autour de safe_save de tinygrad avec retry sur Windows
    """
    path = pathlib.Path(fn)
    
    for attempt in range(max_retries):
        try:
            # Appeler la vraie fonction safe_save de tinygrad
            tinygrad_safe_save(tensors, fn)
            
            
            # Vérifier que le fichier existe et n'est pas vide
            if path.exists() and path.stat().st_size > 100:
                return  # Succès !
            else:
                raise IOError("Fichier invalide après sauvegarde")
                
        except (PermissionError, IOError) as e:
                # Dernier recours : timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_fn = str(fn).replace('.safetensors', f'_{timestamp}.safetensors')
                print(f"\nSauvegarde : {pathlib.Path(new_fn).name}")
                tinygrad_safe_save(tensors, new_fn)
                return