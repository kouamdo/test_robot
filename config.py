#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fichier de configuration pour la pratique SORT.
Centralise tous les chemins et paramètres.
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS DES DOSSIERS
# =============================================================================

# Dossier de base du projet (là où se trouve config.py)
BASE_DIR = Path(__file__).parent.absolute()

# Dossier contenant les images (à adapter selon votre structure)
# Si vos images sont dans un sous-dossier "ПР2_data"
DATA_PATH = BASE_DIR / "ПР2_data"

# Si vos images sont directement dans le dossier principal, décommentez:
# DATA_PATH = BASE_DIR

# Chemin vers le modèle YOLO pré-entraîné
MODEL_PATH = BASE_DIR / "mega_hodov2.pt"

# Dossier de sortie pour les résultats
OUTPUT_DIR = BASE_DIR / "output"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
FRAMES_DIR = OUTPUT_DIR / "frames"

# =============================================================================
# PARAMÈTRES DE DÉTECTION YOLO
# =============================================================================

# Classe cible (4 = "букса" d'après le sujet)
CLASS_ID = 4

# Seuil de confiance pour les détections
CONF_THRESHOLD = 0.5

# =============================================================================
# PARAMÈTRES DU TRACKER SORT
# =============================================================================

# Nombre maximum de frames sans détection avant suppression d'un track
MAX_AGE = 5

# Nombre minimum de détections pour confirmer un track
MIN_HITS = 3

# Seuil IoU pour l'association détection-track
IOU_THRESHOLD = 0.3

# =============================================================================
# PARAMÈTRES DE TRAITEMENT
# =============================================================================

# Nombre d'images à traiter (None = toutes, utile pour les tests)
# Exemple: LIMIT_FRAMES = 100 pour tester rapidement
LIMIT_FRAMES = None

# Afficher les barres de progression
USE_PROGRESS_BAR = True

# Sauvegarder les visualisations intermédiaires
SAVE_INTERMEDIATE_FRAMES = True

# Fréquence de sauvegarde des frames (toutes les N frames)
SAVE_FRAME_INTERVAL = 50

# =============================================================================
# CRÉATION AUTOMATIQUE DES DOSSIERS
# =============================================================================

def create_directories():
    """Crée tous les dossiers nécessaires s'ils n'existent pas."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)
    FRAMES_DIR.mkdir(exist_ok=True)
    print(f"✓ Dossiers créés dans: {OUTPUT_DIR}")

# =============================================================================
# VÉRIFICATION DES CHEMINS
# =============================================================================

def check_paths():
    """Vérifie que tous les chemins nécessaires existent."""
    issues = []
    
    if not DATA_PATH.exists():
        issues.append(f"❌ Dossier images introuvable: {DATA_PATH}")
    else:
        # Vérifier qu'il y a des images
        images = list(DATA_PATH.glob("*.png"))
        if not images:
            issues.append(f"❌ Aucune image .png trouvée dans {DATA_PATH}")
        else:
            print(f"✓ Dossier images trouvé avec {len(images)} images")
    
    if not MODEL_PATH.exists():
        issues.append(f"❌ Fichier modèle introuvable: {MODEL_PATH}")
    else:
        print(f"✓ Modèle YOLO trouvé: {MODEL_PATH.name}")
    
    if issues:
        print("\n".join(issues))
        return False
    return True

# =============================================================================
# AFFICHAGE DE LA CONFIGURATION
# =============================================================================

def print_config():
    """Affiche la configuration actuelle."""
    print("="*60)
    print("CONFIGURATION DU PROJET")
    print("="*60)
    print(f"Dossier de base: {BASE_DIR}")
    print(f"Dossier images: {DATA_PATH}")
    print(f"Modèle YOLO: {MODEL_PATH}")
    print(f"Dossier sortie: {OUTPUT_DIR}")
    print("-"*40)
    print("PARAMÈTRES:")
    print(f"  Classe cible: {CLASS_ID} (букса)")
    print(f"  Seuil confiance YOLO: {CONF_THRESHOLD}")
    print(f"  max_age: {MAX_AGE}")
    print(f"  min_hits: {MIN_HITS}")
    print(f"  iou_threshold: {IOU_THRESHOLD}")
    print(f"  Limite frames: {LIMIT_FRAMES if LIMIT_FRAMES else 'Toutes'}")
    print("="*60)

# =============================================================================
# EXÉCUTION AU CHARGEMENT
# =============================================================================

# Créer les dossiers automatiquement
create_directories()

# Vérifier les chemins (optionnel - peut être commenté si trop verbeux)
# check_paths()

print("✓ Configuration chargée depuis config.py")