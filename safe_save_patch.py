"""
Patch pour résoudre les problèmes de PermissionError avec safe_save sur Windows
en réessayant plusieurs fois.
"""
import time
import pathlib
from typing import Dict
from tinygrad import Tensor
from tinygrad.nn.state import safe_save as tinygrad_safe_save

def safe_save_windows(tensors: Dict[str, Tensor], fn, max_retries: int = 5) -> None:
    """
    Wrapper autour de safe_save de tinygrad avec une logique de retry pour Windows.
    """
    path = pathlib.Path(fn)
    
    for attempt in range(max_retries):
        try:
            # Appeler la vraie fonction safe_save de tinygrad
            tinygrad_safe_save(tensors, fn)
            
            # Vérifier que le fichier existe et n'est pas vide (optionnel mais bien)
            if path.exists() and path.stat().st_size > 100:
                return  # Succès !
            else:
                raise IOError(f"Fichier sauvegardé invalide ou vide : {fn}")
                
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"\nAvertissement: PermissionError lors de la sauvegarde (tentative {attempt + 1}/{max_retries}). Nouvelle tentative dans 0.5s... Erreur: {e}")
                time.sleep(0.5) # Attendre que le système déverrouille le fichier
            else:
                print(f"\nErreur: Echec critique de la sauvegarde après {max_retries} tentatives.")
                raise e # Lève l'erreur si on a épuisé les tentatives
        
        except IOError as e:
            # Gérer le cas du "Fichier invalide"
            if attempt < max_retries - 1:
                print(f"\nAvertissement: IOError lors de la sauvegarde (tentative {attempt + 1}/{max_retries}). Nouvelle tentative dans 0.5s... Erreur: {e}")
                time.sleep(0.5)
            else:
                print(f"\nErreur: Echec critique de la sauvegarde (IOError) après {max_retries} tentatives.")
                raise e