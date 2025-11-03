# 🧪 Journal d'Expérimentation des Hyperparamètres

Ceci documente les tests effectués pour trouver les meilleurs hyperparamètres pour les modèles MLP et CNN, comme requis par la section 1.2 du projet.

## 1. Modèle MLP (`mnist_mlp.py`)

**Objectif de Précision :** $\ge 95\%$
**Paramètres de base :** `STEPS=150`, `LR=0.02`, `BATCH=512`, `ANGLE=15`, `SCALE=0.1`, `SHIFT=0.1`

| Essai | `STEPS` | `LR` | `BATCH` | `ANGLE` | `SCALE` | Précision Finale | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 150 | 0.02 | 512 | 15 | 0.1 | % | Test de base. |
| **2** | 400 | 0.02 | 512 | 15 | 0.1 | % | Test 1, plus long. |
| **3** | 400 | 0.01 | 512 | 15 | 0.1 | % | LR divisé par 2. |
| **4** | 400 | 0.005| 512 | 15 | 0.1 | % | LR divisé par 4. |
| **5** | 400 | 0.01 | 256 | 15 | 0.1 | % | (En supposant 0.01 meilleur LR) Test BATCH plus petit. |
| **6** | 400 | 0.01 | 128 | 15 | 0.1 | % | Test BATCH très petit. |
| **7** | 400 | 0.01 | 256 | 0  | 0.0 | % | (En supposant 0.01/256 optimal) Test sans augmentation. |
| **8** | 400 | 0.01 | 256 | 20 | 0.15| % | Test avec plus d'augmentation. |
| **9** | 800 | 0.01 | 256 | 15 | 0.1 | % | Entraînement long (meilleure combinaison). |
| **10**| 1000| 0.008| 256 | 15 | 0.1 | % | Entraînement long + affinage du LR. |

**Modèle Final Choisi (MLP) :**
* **Commande :** `STEPS=... ;LR=... ;BATCH=... ;ANGLE=...;SCALE=...;python mnist_mlp.py`
* **Précision :** **XX.XX%**

---

## 2. Modèle CNN (`mnist_convnet.py`)

**Objectif de Précision :** $\ge 98\%$
**Paramètres de base :** `STEPS=150`, `LR=0.02`, `BATCH=512`, `ANGLE=15`, `SCALE=0.1`, `SHIFT=0.1`

| Essai | `STEPS` | `LR` | `BATCH` | `ANGLE` | `SCALE` | Précision Finale | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 150 | 0.02 | 512 | 15 | 0.1 | % | Test de base. |
| **2** | 400 | 0.02 | 512 | 15 | 0.1 | % | Test 1, plus long. |
| **3** | 400 | 0.01 | 512 | 15 | 0.1 | % | LR divisé par 2. |
| **4** | 400 | 0.005| 512 | 15 | 0.1 | % | LR divisé par 4. |
| **5** | 400 | 0.005| 256 | 15 | 0.1 | % | (En supposant 0.005 meilleur LR) Test BATCH plus petit. |
| **6** | 400 | 0.005| 128 | 15 | 0.1 | % | Test BATCH très petit. |
| **7** | 400 | 0.005| 256 | 0  | 0.0 | % | (En supposant 0.005/256 optimal) Test **sans augmentation**. |
| **8** | 400 | 0.005| 256 | 20 | 0.15| % | Test avec **plus** d'augmentation. |
| **9** | 800 | 0.005| 256 | 15 | 0.1 | % | Entraînement long (meilleure combinaison). |
| **10**| 1000| 0.005| 128 | 15 | 0.1 | % | Entraînement long + BATCH plus petit. |

**Modèle Final Choisi (CNN) :**
* **Commande :** `STEPS=... ;LR=... ;BATCH=... ;ANGLE=...;SCALE=...;python mnist_convnet.py`
* **Précision :** **XX.XX%**