#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de chargement des données pour la pratique SORT.
Gère le chargement et le tri des images en utilisant Pillow (PIL).
"""

import os
import numpy as np
from pathlib import Path
import re
from PIL import Image

# Importer la configuration
import config

# =============================================================================
# FONCTIONS DE CHARGEMENT DES IMAGES
# =============================================================================

def get_image_files(data_path=None):
    """
    Récupère et trie tous les fichiers image .png par ordre chronologique.
    
    Args:
        data_path: Chemin vers le dossier d'images (utilise config.DATA_PATH par défaut)
    
    Returns:
        list: Liste des noms de fichiers triés
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    # Récupérer tous les fichiers .png
    all_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_files:
        print(f"⚠️ Aucune image trouvée dans {data_path}")
        return []
    
    # Fonction pour extraire le numéro de frame du nom de fichier
    def extract_frame_number(filename):
        # Cherche un nombre dans le nom du fichier (ex: rgb_175.png -> 175)
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])  # Prend le dernier nombre trouvé
        return 0
    
    # Trier par numéro de frame
    sorted_files = sorted(all_files, key=extract_frame_number)
    
    print(f"📸 {len(sorted_files)} images trouvées dans {data_path}")
    if sorted_files:
        print(f"   Première: {sorted_files[0]} (frame {extract_frame_number(sorted_files[0])})")
        print(f"   Dernière: {sorted_files[-1]} (frame {extract_frame_number(sorted_files[-1])})")
    
    return sorted_files


def load_image(filename, data_path=None):
    """
    Charge une image depuis le disque avec Pillow.
    
    Args:
        filename: Nom du fichier image
        data_path: Chemin vers le dossier d'images (utilise config.DATA_PATH par défaut)
    
    Returns:
        numpy.ndarray: Image chargée au format RGB (H, W, 3)
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    img_path = data_path / filename if isinstance(data_path, Path) else Path(data_path) / filename
    
    if not img_path.exists():
        raise FileNotFoundError(f"❌ Image non trouvée: {img_path}")
    
    # Charger l'image avec Pillow
    try:
        pil_image = Image.open(img_path)
        
        # Convertir en RGB si nécessaire
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convertir en numpy array
        img = np.array(pil_image)
        
        return img
    
    except Exception as e:
        raise ValueError(f"❌ Impossible de lire l'image {img_path}: {e}")


def load_images_batch(filenames, data_path=None):
    """
    Charge un lot d'images en une seule fois.
    
    Args:
        filenames: Liste des noms de fichiers
        data_path: Chemin vers le dossier d'images
    
    Returns:
        list: Liste des images chargées
    """
    images = []
    for filename in filenames:
        try:
            img = load_image(filename, data_path)
            images.append(img)
        except Exception as e:
            print(f"⚠️ Erreur avec {filename}: {e}")
    
    return images


# =============================================================================
# FONCTIONS D'INFORMATION SUR LES IMAGES
# =============================================================================

def get_image_shape(filename, data_path=None):
    """
    Retourne les dimensions d'une image sans la charger complètement.
    
    Args:
        filename: Nom du fichier image
        data_path: Chemin vers le dossier d'images
    
    Returns:
        tuple: (hauteur, largeur, canaux)
    """
    if data_path is None:
        data_path = config.DATA_PATH
    
    img_path = data_path / filename if isinstance(data_path, Path) else Path(data_path) / filename
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            mode = img.mode
            channels = 3 if mode == 'RGB' else (1 if mode == 'L' else 4)
            return (height, width, channels)
    except Exception:
        return None


def get_frame_numbers(image_files):
    """
    Extrait les numéros de frame de tous les fichiers.
    
    Args:
        image_files: Liste des noms de fichiers
    
    Returns:
        list: Liste des numéros de frame
    """
    frame_numbers = []
    for f in image_files:
        numbers = re.findall(r'\d+', f)
        if numbers:
            frame_numbers.append(int(numbers[-1]))
    return frame_numbers


def get_image_stats(image_files, data_path=None):
    """
    Calcule des statistiques sur les images (tailles, etc.)
    
    Args:
        image_files: Liste des noms de fichiers
        data_path: Chemin vers le dossier d'images
    
    Returns:
        dict: Statistiques des images
    """
    if not image_files:
        return {}
    
    # Prendre un échantillon pour les stats (pour éviter de tout charger)
    sample_size = min(10, len(image_files))
    sample_files = image_files[:sample_size]
    
    heights = []
    widths = []
    channels = []
    file_sizes = []
    
    for f in sample_files:
        img_path = (data_path or config.DATA_PATH) / f
        shape = get_image_shape(f, data_path)
        if shape:
            h, w, c = shape
            heights.append(h)
            widths.append(w)
            channels.append(c)
        
        # Taille du fichier
        if img_path.exists():
            file_sizes.append(img_path.stat().st_size)
    
    stats = {
        'count': len(image_files),
        'height_mean': np.mean(heights) if heights else 0,
        'height_std': np.std(heights) if heights else 0,
        'width_mean': np.mean(widths) if widths else 0,
        'width_std': np.std(widths) if widths else 0,
        'channels': channels[0] if channels else 3,
        'first_frame': get_frame_numbers([image_files[0]])[0] if image_files else 0,
        'last_frame': get_frame_numbers([image_files[-1]])[0] if image_files else 0,
        'avg_file_size_kb': np.mean(file_sizes) / 1024 if file_sizes else 0
    }
    
    return stats


# =============================================================================
# FONCTIONS DE CONVERSION
# =============================================================================

def image_to_pil(image_array):
    """
    Convertit un numpy array en image PIL.
    
    Args:
        image_array: numpy array (H, W, 3)
    
    Returns:
        PIL.Image: Image PIL
    """
    return Image.fromarray(image_array)


def pil_to_image(pil_image):
    """
    Convertit une image PIL en numpy array.
    
    Args:
        pil_image: PIL.Image
    
    Returns:
        numpy.ndarray: Image au format RGB
    """
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return np.array(pil_image)


# =============================================================================
# FONCTIONS DE PRÉTRAITEMENT
# =============================================================================

def resize_image(image, target_size=(640, 640)):
    """
    Redimensionne une image avec Pillow.
    
    Args:
        image: numpy array ou PIL.Image
        target_size: (largeur, hauteur) cible
    
    Returns:
        numpy.ndarray: Image redimensionnée
    """
    # Convertir en PIL si c'est un numpy array
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Redimensionner avec Pillow (conserve le ratio)
    pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Créer une nouvelle image avec la taille exacte (fond noir)
    new_image = Image.new('RGB', target_size, (0, 0, 0))
    
    # Coller l'image redimensionnée au centre
    paste_x = (target_size[0] - pil_image.size[0]) // 2
    paste_y = (target_size[1] - pil_image.size[1]) // 2
    new_image.paste(pil_image, (paste_x, paste_y))
    
    return np.array(new_image)


def save_image(image, filename, output_dir=None):
    """
    Sauvegarde une image avec Pillow.
    
    Args:
        image: numpy array ou PIL.Image
        filename: Nom du fichier de sortie
        output_dir: Dossier de sortie (utilise config.OUTPUT_DIR par défaut)
    
    Returns:
        Path: Chemin du fichier sauvegardé
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / filename
    
    # Convertir en PIL si c'est un numpy array
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Sauvegarder
    pil_image.save(output_path)
    
    return output_path


# =============================================================================
# FONCTION PRINCIPALE DE TEST
# =============================================================================

def test_data_loader():
    """Teste les fonctions du module avec Pillow."""
    print("="*50)
    print("TEST DU MODULE DATA_LOADER (avec Pillow)")
    print("="*50)
    
    # Vérifier que Pillow est installé
    try:
        from PIL import Image, __version__
        print(f"✅ Pillow version: {__version__}")
    except ImportError:
        print("❌ Pillow n'est pas installé. Installez-le avec: pip install pillow")
        return
    
    # Tester la récupération des fichiers
    print("\n1. Récupération des fichiers:")
    image_files = get_image_files()
    
    if not image_files:
        print("❌ Aucune image trouvée - vérifiez le chemin")
        return
    
    # Tester le chargement d'une image
    print("\n2. Chargement d'une image:")
    first_image = image_files[0]
    img = load_image(first_image)
    print(f"   Image chargée: {first_image}")
    print(f"   Dimensions: {img.shape}")
    print(f"   Type: {img.dtype}")
    print(f"   Min/Max valeurs: {img.min()}/{img.max()}")
    
    # Tester les statistiques
    print("\n3. Statistiques:")
    stats = get_image_stats(image_files[:5])
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Tester les numéros de frame
    print("\n4. Numéros de frame:")
    frame_nums = get_frame_numbers(image_files[:5])
    print(f"   Échantillon: {frame_nums}")
    
    # Tester le redimensionnement
    print("\n5. Test de redimensionnement:")
    resized = resize_image(img, target_size=(416, 416))
    print(f"   Original: {img.shape} → Redimensionné: {resized.shape}")
    
    # Tester la sauvegarde
    print("\n6. Test de sauvegarde:")
    saved_path = save_image(resized, "test_resized.png")
    print(f"   Image sauvegardée: {saved_path}")
    
    print("\n✅ Tests terminés avec succès!")


# =============================================================================
# EXÉCUTION SI LE FICHIER EST LANCÉ DIRECTEMENT
# =============================================================================

if __name__ == "__main__":
    test_data_loader()