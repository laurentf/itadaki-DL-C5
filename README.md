# 🍜 Itadaki - Recipe Image Retrieval System

_"Itadaki" signifie "bon appétit" en japonais_

## 🎓 Projet de certification DEV IA - Alyra

Ce projet constitue le **projet final** de la certification **Développeur Intelligence Artificielle** d'Alyra. Il implémente un système de recherche d'images de recettes utilisant l'apprentissage profond.

**Compétences évaluées (Bloc C5)** :

- **C1** : Préparer des données non structurées en les convertissant en données numériques
- **C2** : Sélectionner l'algorithme d'apprentissage profond le plus adapté
- **C3** : Entraîner un modèle d'apprentissage profond en optimisant une loss function
- **C4** : Déployer efficacement un modèle d'apprentissage profond

## 🎯 Vue d'ensemble

Système d'intelligence artificielle qui **trouve des recettes similaires** à partir d'images de nourriture. Il utilise EfficientNetB0 et l'apprentissage profond pour analyser des photos de plats et proposer des recettes correspondantes depuis une base de données de 13,000+ recettes.

## 📚 Évolution du projet - 4 notebooks progressifs

Ce projet documente une **approche itérative** d'amélioration continue, avec 4 notebooks montrant l'évolution des techniques et des performances :

### 1. 🔸 Raw Model (`recipe_image_retrieval_raw.ipynb`)

**Premier essai - EfficientNet gelé**

- **Approche** : Utilisation d'EfficientNetB0 pré-entraîné **sans modification**
- **Architecture** : `EfficientNetB0 (frozen) → GlobalAveragePooling → Features (1280D)`
- **Technique** : Extraction de features natives sans entraînement supplémentaire
- **Avantages** : Rapide à implementer, baseline solide
- **Limites** : Pas optimisé pour la similarité de recettes
- **Temps** : ~5 minutes setup

### 2. 🔥 Transfer Learning Simple (`recipe_image_retrieval_tl.ipynb`)

**Deuxième approche - Transfer Learning avec sampling aléatoire**

- **Approche** : Triplet Loss avec EfficientNet gelé + tête personnalisée
- **Architecture** : `EfficientNetB0 (frozen) → Custom Head (1024→512) → L2 Norm`
- **Technique** :
  - **Random sampling** pour les triplets (anchor, positive, negative)
  - **Triplet Loss** avec margin = 0.3
  - Métrique : `triplet_accuracy` (pos_sim > neg_sim)
- **Amélioration** : Embeddings optimisés pour la similarité de recettes
- **Limite** : Sampling aléatoire pas optimal, métrique peu informative
- **Temps** : ~30-45 minutes

### 3. ⚡ Transfer Learning Amélioré (`recipe_image_retrieval_tl_hard.ipynb`)

**Troisième approche - Negative Sampling**

- **Approche** : Transfer Learning avec **hard negative mining**
- **Architecture** : Identique au TL simple mais avec sampling intelligent
- **Technique** :
  - **Hard negative sampling** : sélection des négatifs les plus difficiles
  - **Triplet margin accuracy** : mesure si `pos_sim - neg_sim > margin`
  - Amélioration de l'efficacité d'entraînement
- **Avantages** :
  - Apprentissage plus efficace avec exemples difficiles
  - Métrique plus informative (respect du margin)
  - Convergence plus rapide
- **Temps** : ~30-45 minutes (mais plus efficace)

### 4. 🚀 Fine-tuning 2 Phases (`recipe_image_retrieval_ft_hard.ipynb`)

**Quatrième approche - Fine-tuning en 2 phases**

- **Approche** : Fine-tuning en **2 phases** avec hard negative sampling
- **Architecture** :
  - **Phase 1** : Transfer Learning (EfficientNet gelé + Custom Head)
  - **Phase 2** : Fine-tuning (40 couches EfficientNet dégelées)
- **Technique** :
  - **Phase 1** : Entraînement de la tête personnalisée uniquement
  - **Phase 2** : Dégelage intelligent de 40 couches avec learning rates différentiés
  - **Hard negative sampling** pour optimiser l'apprentissage
  - **Triplet margin accuracy** pour un suivi précis
- **Avantages** :
  - Adaptation fine du backbone EfficientNet
  - Performances maximales
  - Entraînement stable et contrôlé
- **Temps** : ~1-2 heures

## 🔬 Comparaison des techniques et performances attendues

| Notebook        | Technique        | Sampling      | Métrique                | Backbone            | Temps  | Performance   |
| --------------- | ---------------- | ------------- | ----------------------- | ------------------- | ------ | ------------- |
| **Raw**         | Features natives | -             | Similarité cosinus      | Gelé                | 5 min  | 🟡 Baseline   |
| **TL Simple**   | Triplet Loss     | Random        | triplet_accuracy        | Gelé                | 45 min | 🟢 Bonne      |
| **TL Hard**     | Triplet Loss     | Hard negative | triplet_margin_accuracy | Gelé                | 45 min | 🟢 Très bonne |
| **FT 2 Phases** | Fine-tuning      | Hard negative | triplet_margin_accuracy | 40 couches dégelées | 2h     | 🟢 Excellente |

## 🛠️ Installation

### Prérequis

- **Python 3.12+** installé
- **Au moins 8GB de RAM** (recommandé: 16GB+)
- **Espace disque** : ~5GB pour les données et modèles
- **GPU recommandé** pour le fine-tuning

### Installation rapide

```bash
# 1. Cloner le projet
git clone [your-repo-url]
cd itadaki

# 2. Créer l'environnement virtuel
python -m venv itadaki_env

# 3. Activer l'environnement
# Windows:
itadaki_env\Scripts\activate
# Linux/Mac:
source itadaki_env/bin/activate

# 4. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 5. Lancer Jupyter
jupyter notebook
```

## 📊 Structure du Projet

```
itadaki/
├── 📓 NOTEBOOKS (Evolution Progressive)
│   ├── recipe_image_retrieval_raw.ipynb         # ✅ 1. EfficientNet gelé
│   ├── recipe_image_retrieval_tl.ipynb          # ✅ 2. TL + Random sampling
│   ├── recipe_image_retrieval_tl_hard.ipynb     # ✅ 3. TL + Hard negative sampling
│   └── recipe_image_retrieval_ft_hard.ipynb     # ✅ 4. Fine-tuning 2 phases
├── 📁 MODÈLES & DONNÉES ENTRAÎNÉS
│   ├── raw/                                     # 🔸 Raw Model (EfficientNet gelé)
│   │   ├── recipe_image_retrieval_model_raw.keras        # Modèle embeddings
│   │   ├── recipe_embeddings_database_raw.npy            # Base embeddings (1280D)
│   │   └── recipe_embeddings_database_metadata_raw.pkl   # Metadata pour recherche
│   ├── tl/                                      # 🔥 Transfer Learning
│   │   ├── best_embedding_recipe_image_retrieval_model_tl.keras  # Modèle embeddings
│   │   ├── recipe_embeddings_database_tl.npy                     # Base embeddings (512D)
│   │   └── recipe_embeddings_database_metadata_tl.pkl            # Metadata pour recherche
│   ├── ft/                                      # 🚀 Fine-tuning 2 Phases
│   │   ├── best_embedding_recipe_image_retrieval_model_ft.keras  # Modèle embeddings
│   │   ├── recipe_embeddings_database_ft.npy                     # Base embeddings (512D)
│   │   └── recipe_embeddings_database_metadata_ft.pkl            # Metadata pour recherche
│   └── data/                                    # 📊 Dataset principal
│       ├── recipes_with_images_dataframe.pkl   # DataFrame pickle (rechargement facile)
│       └── data.csv                             # Données CSV originales
├── 🖼️ TESTS
│   └── test_recipes/                            # Images de test variées
├── 📊 RAPPORTS
│   └── reports/                                 # Analyses et visualisations
└── 🔧 CONFIGURATION
    ├── requirements.txt
    └── README.md
```

### 🔍 Détail des dossiers de modèles

Chaque dossier de modèle (`raw/`, `tl/`, `ft/`) contient **3 fichiers essentiels** :

#### 📦 **Modèle Embeddings** (`.keras`)

- **Raw** : `recipe_image_retrieval_model_raw.keras` - EfficientNet gelé (1280D)
- **TL** : `best_embedding_recipe_image_retrieval_model_tl.keras` - TL optimisé (512D)
- **FT** : `best_embedding_recipe_image_retrieval_model_ft.keras` - Fine-tuning 2 phases (512D)

#### 🗄️ **Base de données d'embeddings** (`.npy`)

- **Raw** : `recipe_embeddings_database_raw.npy` - 13,463 embeddings × 1280D
- **TL** : `recipe_embeddings_database_tl.npy` - 13,463 embeddings × 512D
- **FT** : `recipe_embeddings_database_ft.npy` - 13,463 embeddings × 512D

#### 📋 **Metadata** (`.pkl`)

- **Raw** : `recipe_embeddings_database_metadata_raw.pkl` - Index → recette mapping
- **TL** : `recipe_embeddings_database_metadata_tl.pkl` - Index → recette mapping
- **FT** : `recipe_embeddings_database_metadata_ft.pkl` - Index → recette mapping

#### 💾 **Dataset principal** (`data/`)

- **`recipes_with_images_dataframe.pkl`** : DataFrame pickle complet pour rechargement rapide
- **`data.csv`** : Données CSV originales du dataset Kaggle

## 🚀 Guide d'utilisation

### Approche Recommandée : Progression Séquentielle

Pour comprendre l'évolution du projet, il est recommandé de suivre les notebooks dans l'ordre :

#### 1. Commencer par le Raw Model

```bash
jupyter notebook recipe_image_retrieval_raw.ipynb
```

- Comprendre la baseline et l'extraction de features
- Tester rapidement le système

#### 2. Continuer avec Transfer Learning Simple

```bash
jupyter notebook recipe_image_retrieval_tl.ipynb
```

- Découvrir le Triplet Loss et l'optimisation d'embeddings
- Voir l'amélioration par rapport au raw model

#### 3. Améliorer avec Hard Negative Sampling

```bash
jupyter notebook recipe_image_retrieval_tl_hard.ipynb
```

- Comprendre l'importance du sampling intelligent
- Observer l'amélioration de l'efficacité d'entraînement

#### 4. Finaliser avec Fine-tuning 2 Phases

```bash
jupyter notebook recipe_image_retrieval_ft_hard.ipynb
```

- Découvrir le fine-tuning sophistiqué
- Obtenir les meilleures performances

## 🎯 Utilisation du Système

### Interface Commune à tous les Notebooks

```python
# 1. Charger votre image
query_image = "path/to/your/food_image.jpg"

# 2. Rechercher les recettes similaires
results = retrieval_system.search_similar_recipes(query_image, top_k=3)

# 3. Afficher les résultats avec visualisations
retrieval_system.display_results(query_image, results)
```

### Fonctionnalités Avancées

```python
# Visualiser l'architecture du modèle
retrieval_system.visualize_model_architecture()

# Analyser les triplets d'entraînement
retrieval_system.show_triplets(num_triplets=3)

# Évaluer les performances
retrieval_system.evaluate_model()
```

## 📥 Dataset

**Food Ingredients and Recipe Dataset with Images** (Kaggle)

🔗 **Lien Kaggle** : https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

- **13,463 recettes uniques** avec images HD
- **Ingrédients détaillés** et instructions complètes
- **Images haute qualité** (224x224 minimum)
- **Téléchargement automatique** via `kagglehub`
- **Taille totale** : ~2GB

## 🔧 Architecture Technique Détaillée

### Configuration par Notebook

#### Raw Model

```python
CONFIG_RAW = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EMBEDDING_DIM': 1280,  # Features natives EfficientNet
}
```

#### Transfer Learning Simple

```python
CONFIG_TL = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EMBEDDING_DIM': 512,
    'TRIPLET_MARGIN': 0.3,
    'EPOCHS': 10,
    'SAMPLING': 'random'  # Random triplet sampling
}
```

#### Transfer Learning Hard

```python
CONFIG_TL_HARD = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EMBEDDING_DIM': 512,
    'TRIPLET_MARGIN': 0.8,
    'EPOCHS': 15,
    'SAMPLING': 'hard_negative'  # Hard negative mining
}
```

#### Fine-tuning 2 Phases

```python
CONFIG_FT = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EMBEDDING_DIM': 512,
    'TRIPLET_MARGIN': 0.8,
    'LAYERS_TO_UNFREEZE': 40,
    'PHASE_1_EPOCHS': 6,  # Transfer Learning
    'PHASE_2_EPOCHS': 20,  # Fine-tuning
    'SAMPLING': 'hard_negative'
}
```

## 📈 Évolution des Performances

### Métriques Clés

| Notebook    | Métrique Principale     | Dimension | Sampling      | Temps Train |
| ----------- | ----------------------- | --------- | ------------- | ----------- |
| Raw         | N/A                     | 1280D     | N/A           | N/A         |
| TL Simple   | triplet_accuracy        | 512D      | Random        | 30 min      |
| TL Hard     | triplet_margin_accuracy | 512D      | Hard negative | 35 min      |
| FT 2 Phases | triplet_margin_accuracy | 512D      | Hard negative | 90 min      |

### Amélioration Progressive

1. **Raw → TL Simple** : Embeddings optimisés pour la similarité
2. **TL Simple → TL Hard** : Sampling intelligent + métrique plus précise
3. **TL Hard → FT 2 Phases** : Adaptation fine du backbone pour performances maximales

## 🖼️ Images de Test

Le dossier `test_recipes/` contient des images variées pour tester les différents modèles :

```python
# Exemple d'utilisation
test_image = "./test_recipes/fraisier-matcha.jpg"
results = retrieval_system.search_similar_recipes(test_image, top_k=5)
```

## 🔍 Concepts clés implémentés

### 1. Triplet Loss

Optimisation de la distance entre embeddings pour maximiser la similarité intra-classe et minimiser la similarité inter-classe.

### 2. Hard Negative Mining

Sélection intelligente des exemples négatifs les plus difficiles pour améliorer l'efficacité d'entraînement.

### 3. Fine-tuning 2 Phases

Approche progressive : d'abord entraîner la tête personnalisée, puis adapter le backbone pré-entraîné.

### 4. Triplet Margin Accuracy

Métrique avancée qui mesure si la différence `pos_similarity - neg_similarity > margin`, plus informative que la Triplet Accuracy.

## 🎯 Résultats et Apprentissages

### Principales Découvertes

1. **Raw Model** : Baseline solide mais non optimisée
2. **Random Sampling** : Efficace mais sous-optimal
3. **Hard Negative Mining** : Amélioration significative de l'efficacité
4. **Fine-tuning 2 Phases** : Performances maximales avec contrôle total

### Recommandations

- **Pour tests rapides** : Utiliser le Raw Model
- **Pour production** : Transfer Learning Hard est le meilleur compromis
- **Pour recherche avancée** : Fine-tuning 2 Phases pour performances maximales

## 📚 Technologies Utilisées

- **TensorFlow/Keras** : Framework d'apprentissage profond
- **EfficientNetB0** : Architecture de backbone
- **OpenCV** : Traitement d'images
- **Matplotlib/Seaborn** : Visualisations
- **NumPy/Pandas** : Manipulation de données
- **Kagglehub** : Téléchargement de dataset

## 🎓 Conclusion

Ce projet démontre une approche méthodique d'amélioration continue en intelligence artificielle, depuis une baseline simple jusqu'à des techniques avancées de fine-tuning. Chaque notebook apporte des améliorations progressives et documente les apprentissages obtenus.

La progression **Raw → TL Simple → TL Hard → FT 2 Phases** illustre parfaitement comment optimiser graduellement un système d'apprentissage profond pour obtenir des performances maximales.

---

**Projet réalisé dans le cadre de la certification pour la formation de Développeur IA - Alyra**
