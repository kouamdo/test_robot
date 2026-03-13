#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de détection pour la pratique SORT.
Utilise YOLOv8n pour détecter les objets de classe 4 (букса).
"""

import numpy as np
import torch
from ultralytics import YOLO
import time
from pathlib import Path

# Importer la configuration
import config

# =============================================================================
# CLASSE PRINCIPALE DE DÉTECTION
# =============================================================================

class YOLODetector:
    """
    Détecteur YOLO pour les objets de classe 4 (букса).
    """
    
    def __init__(self, model_path=None, class_id=None, conf_threshold=None, device=None):
        """
        Initialise le détecteur YOLO.
        
        Args:
            model_path: Chemin vers le modèle .pt (utilise config.MODEL_PATH par défaut)
            class_id: ID de la classe à détecter (utilise config.CLASS_ID par défaut)
            conf_threshold: Seuil de confiance (utilise config.CONF_THRESHOLD par défaut)
            device: 'cuda' ou 'cpu' (auto-détection si None)
        """
        # Utiliser les valeurs de config si non spécifiées
        self.model_path = model_path if model_path else config.MODEL_PATH
        self.class_id = class_id if class_id is not None else config.CLASS_ID
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        
        # Déterminer le device (GPU si disponible)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🔄 Chargement du modèle YOLO depuis {self.model_path}...")
        print(f"   Device: {self.device}")
        print(f"   Classe cible: {self.class_id}")
        print(f"   Seuil de confiance: {self.conf_threshold}")
        
        # Charger le modèle
        start_time = time.time()
        self.model = YOLO(self.model_path)
        load_time = time.time() - start_time
        
        print(f"✓ Modèle chargé en {load_time:.2f} secondes")
        
        # Statistiques
        self.total_detections = 0
        self.frames_processed = 0
        self.detection_times = []
    
    def detect(self, image, conf_threshold=None):
        """
        Détecte les objets sur une image.
        
        Args:
            image: Image en RGB (numpy array) ou chemin vers l'image
            conf_threshold: Seuil de confiance (utilise celui de l'instance si None)
        
        Returns:
            numpy array: Détections au format [x1, y1, x2, y2, confidence]
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        # Mesurer le temps
        start_time = time.time()
        
        # Détection
        results = self.model(image, conf=conf_threshold, verbose=False)[0]
        
        # Extraire les détections de la classe cible
        detections = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Vérifier la classe
                if int(box.cls) == self.class_id:
                    # Coordonnées
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Confiance
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])
        
        # Mettre à jour les statistiques
        detect_time = time.time() - start_time
        self.detection_times.append(detect_time)
        self.frames_processed += 1
        self.total_detections += len(detections)
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def detect_batch(self, images, conf_threshold=None, batch_size=16):
        """
        Détecte sur un lot d'images (plus rapide).
        
        Args:
            images: Liste d'images ou liste de chemins
            conf_threshold: Seuil de confiance
            batch_size: Taille du lot pour le traitement par lots
        
        Returns:
            list: Liste des détections pour chaque image
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        all_detections = []
        
        # Traiter par lots
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Détection sur le lot
            results = self.model(batch, conf=conf_threshold, verbose=False)
            
            for result in results:
                detections = []
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        if int(box.cls) == self.class_id:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            detections.append([x1, y1, x2, y2, conf])
                
                all_detections.append(np.array(detections) if detections else np.empty((0, 5)))
                self.frames_processed += 1
                self.total_detections += len(detections)
        
        return all_detections
    
    def detect_from_file(self, image_path, conf_threshold=None):
        """
        Détecte sur une image chargée depuis un fichier.
        
        Args:
            image_path: Chemin vers l'image
            conf_threshold: Seuil de confiance
        
        Returns:
            numpy array: Détections
        """
        # Charger l'image
        import cv2
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return self.detect(img_rgb, conf_threshold)
    
    def get_stats(self):
        """
        Retourne les statistiques de détection.
        
        Returns:
            dict: Statistiques
        """
        stats = {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': self.total_detections / max(1, self.frames_processed),
            'avg_detection_time_ms': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'min_detection_time_ms': np.min(self.detection_times) * 1000 if self.detection_times else 0,
            'max_detection_time_ms': np.max(self.detection_times) * 1000 if self.detection_times else 0,
            'device': self.device
        }
        return stats
    
    def reset_stats(self):
        """Réinitialise les statistiques."""
        self.total_detections = 0
        self.frames_processed = 0
        self.detection_times = []


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def filter_detections_by_confidence(detections, threshold):
    """
    Filtre les détections par seuil de confiance.
    
    Args:
        detections: Tableau de détections [x1,y1,x2,y2,conf]
        threshold: Seuil de confiance
    
    Returns:
        numpy array: Détections filtrées
    """
    if len(detections) == 0:
        return detections
    
    mask = detections[:, 4] >= threshold
    return detections[mask]


def filter_detections_by_size(detections, min_area=100, max_area=None):
    """
    Filtre les détections par taille.
    
    Args:
        detections: Tableau de détections [x1,y1,x2,y2,conf]
        min_area: Aire minimale
        max_area: Aire maximale (None = pas de limite)
    
    Returns:
        numpy array: Détections filtrées
    """
    if len(detections) == 0:
        return detections
    
    areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
    mask = areas >= min_area
    
    if max_area is not None:
        mask &= areas <= max_area
    
    return detections[mask]


def convert_detections_to_center_format(detections):
    """
    Convertit les détections du format [x1,y1,x2,y2] vers [cx,cy,w,h].
    
    Args:
        detections: Tableau de détections [x1,y1,x2,y2,conf]
    
    Returns:
        numpy array: Détections au format [cx,cy,w,h,conf]
    """
    if len(detections) == 0:
        return detections
    
    converted = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        converted.append([cx, cy, w, h, conf])
    
    return np.array(converted)


def convert_detections_from_center_format(detections):
    """
    Convertit les détections du format [cx,cy,w,h] vers [x1,y1,x2,y2].
    
    Args:
        detections: Tableau de détections [cx,cy,w,h,conf]
    
    Returns:
        numpy array: Détections au format [x1,y1,x2,y2,conf]
    """
    if len(detections) == 0:
        return detections
    
    converted = []
    for det in detections:
        cx, cy, w, h, conf = det
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        converted.append([x1, y1, x2, y2, conf])
    
    return np.array(converted)


# =============================================================================
# FONCTION PRINCIPALE DE TEST
# =============================================================================

def test_detector():
    """Teste le détecteur sur un échantillon d'images."""
    print("="*50)
    print("TEST DU MODULE DETECTION")
    print("="*50)
    
    # Vérifier que le modèle existe
    if not config.MODEL_PATH.exists():
        print(f"❌ Modèle non trouvé: {config.MODEL_PATH}")
        return
    
    # Initialiser le détecteur
    print("\n1. Initialisation du détecteur:")
    detector = YOLODetector()
    
    # Chercher des images de test
    import data_loader
    image_files = data_loader.get_image_files()
    
    if not image_files:
        print("❌ Aucune image trouvée pour le test")
        return
    
    # Tester sur quelques images
    print("\n2. Test sur 3 images:")
    test_files = image_files[:3]
    
    for i, img_file in enumerate(test_files):
        # Charger l'image
        img = data_loader.load_image(img_file)
        
        # Détecter
        detections = detector.detect(img)
        
        print(f"   Image {i+1}: {img_file}")
        print(f"      Détections: {len(detections)}")
        if len(detections) > 0:
            print(f"      Confiance moyenne: {np.mean(detections[:, 4]):.3f}")
            print(f"      Première boîte: {detections[0, :4].astype(int)}")
    
    # Afficher les statistiques
    print("\n3. Statistiques:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test terminé avec succès!")


# =============================================================================
# EXÉCUTION SI LE FICHIER EST LANCÉ DIRECTEMENT
# =============================================================================

if __name__ == "__main__":
    test_detector()