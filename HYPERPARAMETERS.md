# 🧪 Journal d'Expérimentation des Hyperparamètres

Ceci documente les tests effectués pour trouver les meilleurs hyperparamètres pour les modèles MLP et CNN, comme requis par la section 1.2 du projet.

## 1. Modèle MLP (`mnist_mlp.py`)

**Objectif de Précision :** $\ge 95\%$
**Paramètres de base :** `STEPS=50`, `LR=0.02`, `BATCH=512`, `ANGLE=15`, `SCALE=0.1`, `SHIFT=0.1`

| Essai | `STEPS` | `LR` | `BATCH` | `ANGLE` | `SCALE` | Loss | Précision Finale | Notes | Temps d'entrainement
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 50  | 0.02 | 512 | 15 | 0.1 | 1.83 | 72.18% | Test de base. | 00:09 min
| **0** | 70  | 0.02 | 512 | 15 | 0.1 | 0.52 | 90.49% | Test de base. | 00:09 min
| **1** | 150 | 0.02 | 512 | 15 | 0.1 | 0.16 | 96.20% | STEP * 2. | 00:17 min
| **2** | 400 | 0.02 | 512 | 15 | 0.1 | 0.10 | 97.63% | Test 1, plus long. | 00:37 min
| **3** | 400 | 0.01 | 512 | 15 | 0.1 | 0.15 | 96.82% | LR divisé par 2. | 00:33 min
| **4** | 400 | 0.04 | 512 | 15 | 0.1 | 0.18 | 97.79% | LR multiplié par 2. | 00:37 min
| **5** | 400 | 0.04 | 256 | 15 | 0.1 | 0.20 | 97.22% | (En supposant 0.04 meilleur LR) Test BATCH plus petit. | 00:33 min
| **6** | 400 | 0.04 | 1024| 15 | 0.1 | 0.12 | 98.12% | Test BATCH très grand. | 00:37 min
| **7** | 400 | 0.04 | 1024| 0  | 0.0 | 0.00 | 97.65% | (En supposant 0.02/256 optimal) Test sans augmentation. | 00:37 min
| **8** | 400 | 0.04 | 1024| 20 | 0.15| 0.12 | 97.71%% | Test avec plus d'augmentation. | 00:36 min
| **9** | 800 | 0.04 | 1024| 15 | 0.1 | 0.07 | 98.08% | Entraînement long (meilleure combinaison). | 01:08 min
| **10**| 1000| 0.038| 1024| 15 | 0.1 | 0.07 | 98.22% | Entraînement long + affinage du LR. | 01:23 min
| **11**| 1000| 0.035| 1024| 15 | 0.1 | 0.05 | 98.41% | Entraînement long + affinage du LR. | 01:23 min
| **12**| 1000| 0.03 | 1024| 15 | 0.1 | 0.10 | 98.41% | Entraînement long + affinage du LR. | 01:23 min

**Modèle Final Choisi (MLP) :**
* **Commande :** `$env:STEPS = "1000"; $env:LR = "0.035"; $env:BATCH = "1024";$env:ANGLE = "15"; $env:SCALE = "0.1"; python mnist_mlp.py`
* **Précision :** **98.41%**
* **RAM Used :** **0.18 GB**
* **Layers :** **5 bias**

---

## 2. Modèle CNN (`mnist_convnet.py`)

**Objectif de Précision :** $\ge 98\%$
**Paramètres de base :** `STEPS=10`, `LR=0.02`, `BATCH=512`, `ANGLE=15`, `SCALE=0.1`, `SHIFT=0.1`

| Essai | `STEPS` | `LR` | `BATCH` | `ANGLE` | `SCALE` | `SHIFT` | Précision Finale | Notes | Temps d'entrainement
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 10  | 0.02 | 512 | 15 | 0.1 | 0.1 | 0.89 | 31.18% | Test de base. | 00:24 min
| **1** | 10  | 0.01 | 512 | 15 | 0.1 | 0.1 | 0.85 | 15.34% | LR plus faible. | 00:24 min
| **2** | 10  | 0.04 | 512 | 15 | 0.1 | 0.1 | 0.89 | 23.40% | LR plus élevé. | 00:27 min
| **3** | 50  | 0.02 | 128 | 15 | 0.1 | 0.1 | 0.17 | 97.71% | Plus de STEPS mais moins de BATCH. | 01:08 min
| **4** | 50  | 0.02 | 256 | 15 | 0.1 | 0.1 | 0.13 | 97.09% | Moyen Batch Size. | 01:12 min
| **5** | 50  | 0.04 | 1024| 15 | 0.1 | 0.1 | 0.15 | 95.42% | Grand Batch Size. | 02:37 min
| **6** | 100 | 0.005| 256 | 15 | 0.1 | 0.1 | 0.09 | 98.17% | PTests à 100 Étapes (Impact de l'Augmentation / Sampling). | 01:07 min
| **7** | 100 | 0.02 | 256 | 15 | 0.1 | 0.1 | 0.18 | 97.60% | Test 100 étapes (Sans Augmentation). | 01:07 min
| **8** | 100 | 0.005| 256 | 0  | 0.0 | 0.0 | 0.0  | 89.46% | Test 0 ANGLE, SCALE, SHIFT. | 01:10 min
| **9** | 100 | 0.005| 256 | 30 | 0.2 | 0.2 | 0.20 | 96.71% | Test ANGLE, SCALE, SHIFT doublé. | 01:08 min
| **10**| 100 | 0.005| 256 | 15 | 0.1 | 0.1 | 0.0  | 97.67% | Test 100 étapes (Sampling Bilinear). | 01:07 min
| **11**| 100 | 0.01 | 128 | 15 | 0.1 | 0.1 | 0.26 | 97.19% | Test 100 étapes (Combo agressif : LR plus élevé, petit batch). | 00:45 min

**Modèle Final Choisi (CNN) :**
* **Commande :** `$env:STEPS="100"; $env:LR="0.005"; $env:BATCH="256"; $env:ANGLE="15"; $env:SCALE="0.1"; $env:SHIFT="0.1"; python mnist_convnet.py`
* **Précision :** **98.17%**
* **RAM Used :** **2.21 GB**
* **Layers :** **13 bias**