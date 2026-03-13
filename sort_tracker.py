#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module SORT (Simple Online and Realtime Tracking) pour la pratique.
Implémente le filtrage de Kalman et l'association de données.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import time

# Importer la configuration
import config

# =============================================================================
# FILTRE DE KALMAN POUR LE SUIVI D'UN OBJET
# =============================================================================

class KalmanBoxTracker:
    """
    Cette classe représente le suivi d'un objet avec un filtre de Kalman.
    Chaque objet a son propre filtre.
    """
    
    count = 0  # Compteur statique pour générer des IDs uniques
    
    def __init__(self, bbox):
        """
        Initialise un tracker avec une boîte de détection.
        
        Args:
            bbox: [x1, y1, x2, y2] coordonnées de la boîte
        """
        # Définition du filtre de Kalman avec 7 états et 4 mesures
        # État: [cx, cy, s, r, cx', cy', s']
        #   cx, cy: centre de la boîte
        #   s: surface (width * height)
        #   r: ratio largeur/hauteur (considéré constant)
        #   cx', cy', s': vitesses correspondantes
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Matrice de transition d'état (modèle à vitesse constante)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Matrice de mesure (nous mesurons [cx, cy, s, r])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Matrice de covariance de mesure (incertitude sur les mesures)
        self.kf.R[2:, 2:] *= 10.  # Plus d'incertitude sur s et r
        
        # Matrice de covariance d'état initiale
        self.kf.P[4:, 4:] *= 1000.  # Haute incertitude pour les vitesses
        self.kf.P *= 10.
        
        # Matrice de bruit de processus
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Conversion de [x1,y1,x2,y2] en [cx,cy,s,r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        s = w * h  # surface
        r = w / h  # ratio
        
        # Initialisation de l'état
        self.kf.x[:4] = np.array([cx, cy, s, r])
        
        # Métadonnées du tracker
        self.time_since_update = 0  # Frames depuis dernière mise à jour
        self.id = KalmanBoxTracker.count  # ID unique
        KalmanBoxTracker.count += 1
        
        self.history = []  # Historique des états
        self.hits = 1  # Nombre total de détections associées
        self.hit_streak = 1  # Détections consécutives
        self.age = 0  # Âge du tracker en frames
        
        self.start_frame = None  # Frame de début (sera défini plus tard)
        self.positions = []  # Positions pour l'historique
    
    def update(self, bbox, frame_num=None):
        """
        Met à jour le filtre avec une nouvelle détection.
        
        Args:
            bbox: [x1, y1, x2, y2] nouvelle détection
            frame_num: Numéro de frame (optionnel)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        if frame_num is not None:
            if self.start_frame is None:
                self.start_frame = frame_num
        
        # Conversion en [cx,cy,s,r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        s = w * h
        r = w / h
        
        # Mise à jour du filtre
        self.kf.update(np.array([cx, cy, s, r]))
        
        # Sauvegarder la position
        self.positions.append((cx, cy))
    
    def predict(self):
        """
        Prédit la position suivante du tracker.
        
        Returns:
            np.array: Boîte prédite [x1, y1, x2, y2]
        """
        # Éviter les surfaces négatives
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        # Prédiction
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(self.kf.x)
        
        return self.get_state()
    
    def get_state(self):
        """
        Retourne la boîte prédite au format [x1, y1, x2, y2].
        
        Returns:
            np.array: [x1, y1, x2, y2]
        """
        cx, cy, s, r = self.kf.x[:4]
        
        w = np.sqrt(s * r)
        h = s / w
        
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        return np.array([x1, y1, x2, y2])
    
    def get_center(self):
        """
        Retourne le centre de la boîte.
        
        Returns:
            tuple: (cx, cy)
        """
        return (self.kf.x[0], self.kf.x[1])


# =============================================================================
# FONCTIONS DE CALCUL D'IOU
# =============================================================================

def iou_batch(bb_test, bb_gt):
    """
    Calcule l'IoU (Intersection over Union) entre deux ensembles de boîtes.
    
    Args:
        bb_test: Tableau de boîtes test [N, 4] au format [x1,y1,x2,y2]
        bb_gt: Tableau de boîtes ground truth [M, 4] au format [x1,y1,x2,y2]
    
    Returns:
        np.array: Matrice IoU de taille [N, M]
    """
    # Expand dimensions for broadcasting
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    # Coordonnées de l'intersection
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    # Largeur et hauteur de l'intersection
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # Surfaces des boîtes
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection
    
    # Éviter la division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
        iou = np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    
    return iou


def iou_single(bb_test, bb_gt):
    """
    Calcule l'IoU entre deux boîtes uniques.
    
    Args:
        bb_test: [x1,y1,x2,y2]
        bb_gt: [x1,y1,x2,y2]
    
    Returns:
        float: IoU
    """
    return iou_batch(bb_test.reshape(1, -1), bb_gt.reshape(1, -1))[0, 0]


# =============================================================================
# ALGORITHME SORT PRINCIPAL
# =============================================================================

class Sort:
    """
    Implémentation principale de l'algorithme SORT
    (Simple Online and Realtime Tracking).
    """
    
    def __init__(self, max_age=None, min_hits=None, iou_threshold=None):
        """
        Initialise le tracker SORT.
        
        Args:
            max_age: Nombre de frames sans détection avant suppression
            min_hits: Détections nécessaires pour confirmer un track
            iou_threshold: Seuil IoU pour l'association
        """
        self.max_age = max_age if max_age is not None else config.MAX_AGE
        self.min_hits = min_hits if min_hits is not None else config.MIN_HITS
        self.iou_threshold = iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD
        
        self.trackers = []  # Liste des trackers actifs
        self.frame_count = 0  # Compteur de frames
        self.unique_objects = set()  # IDs uniques détectés
        
        # Statistiques
        self.track_history = defaultdict(list)  # Historique des positions
        self.track_start_frames = {}  # Frame de début pour chaque track
        self.track_end_frames = {}  # Frame de fin pour chaque track
        self.total_tracks_created = 0
        
        print(f"✓ Tracker SORT initialisé:")
        print(f"   max_age: {self.max_age}")
        print(f"   min_hits: {self.min_hits}")
        print(f"   iou_threshold: {self.iou_threshold}")
    
    def update(self, dets, frame_info=None):
        """
        Met à jour les trackers avec les nouvelles détections.
        
        Args:
            dets: Tableau de détections [N, 5] au format [x1,y1,x2,y2,conf]
            frame_info: Informations supplémentaires sur la frame (optionnel)
        
        Returns:
            np.array: Trackers mis à jour [M, 5] au format [x1,y1,x2,y2,id]
        """
        self.frame_count += 1
        
        # Si pas de détections, prédire seulement
        if len(dets) == 0:
            # Prédire pour tous les trackers
            for trk in self.trackers:
                trk.predict()
            
            # Supprimer les trackers trop âgés
            self.trackers = [trk for trk in self.trackers 
                            if trk.time_since_update <= self.max_age]
            
            return np.empty((0, 5))
        
        # Prédiction pour tous les trackers existants
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :4] = pos
            trks[t, 4] = 0  # Score temporaire
            
            # Vérifier les prédictions invalides
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Supprimer les trackers avec prédictions invalides
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)
        
        # Association des détections aux trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)
        
        # Mise à jour des trackers matchés
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                # Trouver la détection associée
                d_idx = matched[np.where(matched[:, 1] == t)[0], 0][0]
                trk.update(dets[d_idx, :4], self.frame_count)
                
                # Mettre à jour l'historique
                center = trk.get_center()
                self.track_history[trk.id].append((self.frame_count, center[0], center[1]))
        
        # Création de nouveaux trackers pour les détections non assignées
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            trk.start_frame = self.frame_count
            self.trackers.append(trk)
            self.total_tracks_created += 1
            
            # Initialiser l'historique
            center = trk.get_center()
            self.track_history[trk.id] = [(self.frame_count, center[0], center[1])]
        
        # Collecte des résultats
        ret = []
        for trk in self.trackers:
            d = trk.get_state()
            
            # Ne garder que les trackers confirmés ou récents
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                self.unique_objects.add(trk.id)
        
        # Suppression des trackers trop âgés
        remaining_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                remaining_trackers.append(trk)
            else:
                # Enregistrer la frame de fin
                self.track_end_frames[trk.id] = self.frame_count
        
        self.trackers = remaining_trackers
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Associe les détections aux trackers existants.
        Utilise l'algorithme hongrois avec la matrice IoU.
        
        Args:
            detections: Tableau de détections [N, 5]
            trackers: Tableau de trackers prédits [M, 5]
        
        Returns:
            tuple: (matched, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2)), np.arange(len(detections)), np.empty((0,))
        
        # Calcul de la matrice IoU
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        # Algorithme hongrois pour minimiser le coût (maximiser IoU)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # Identifier les détections et trackers non matchés
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filtrer les associations avec un IoU trop faible
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def get_count(self):
        """
        Retourne le nombre d'objets uniques comptés.
        
        Returns:
            int: Nombre d'objets uniques
        """
        return len(self.unique_objects)
    
    def get_stats(self):
        """
        Retourne les statistiques du tracker.
        
        Returns:
            dict: Statistiques
        """
        # Calculer la durée de vie moyenne des tracks
        track_lifetimes = []
        for trk_id, history in self.track_history.items():
            if len(history) > 0:
                frames = [h[0] for h in history]
                lifetime = max(frames) - min(frames) + 1
                track_lifetimes.append(lifetime)
        
        stats = {
            'unique_objects': len(self.unique_objects),
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': len(self.trackers),
            'frames_processed': self.frame_count,
            'avg_track_lifetime': np.mean(track_lifetimes) if track_lifetimes else 0,
            'max_track_lifetime': np.max(track_lifetimes) if track_lifetimes else 0,
            'min_track_lifetime': np.min(track_lifetimes) if track_lifetimes else 0
        }
        return stats
    
    def reset(self):
        """
        Réinitialise complètement le tracker.
        """
        self.trackers = []
        self.frame_count = 0
        self.unique_objects = set()
        self.track_history = defaultdict(list)
        self.track_start_frames = {}
        self.track_end_frames = {}
        self.total_tracks_created = 0
        KalmanBoxTracker.count = 0  # Réinitialiser le compteur d'IDs


# =============================================================================
# VERSION AMÉLIORÉE AVEC GESTION ADAPTATIVE
# =============================================================================

class EnhancedSort(Sort):
    """
    Version améliorée de SORT avec:
    - Seuil IoU adaptatif basé sur la densité des détections
    - Gestion de la confiance
    """
    
    def __init__(self, max_age=None, min_hits=None, iou_threshold=None, 
                 adaptive_iou=True, confidence_threshold=0.3):
        """
        Initialise le tracker amélioré.
        
        Args:
            max_age: Âge maximum
            min_hits: Hits minimum pour confirmation
            iou_threshold: Seuil IoU de base
            adaptive_iou: Activer l'adaptation du seuil
            confidence_threshold: Seuil de confiance minimum
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.adaptive_iou = adaptive_iou
        self.confidence_threshold = confidence_threshold
        self.track_confidences = defaultdict(list)
    
    def update(self, dets, frame_info=None):
        """
        Met à jour avec adaptation du seuil.
        """
        # Filtrer les détections par confiance
        if len(dets) > 0 and self.confidence_threshold > 0:
            dets = dets[dets[:, 4] >= self.confidence_threshold]
        
        # Adapter le seuil IoU si demandé
        if self.adaptive_iou and len(dets) > 0:
            # Plus il y a de détections, plus le seuil est élevé
            density = min(1.0, len(dets) / 15.0)
            self.current_iou = self.iou_threshold * (0.8 + 0.4 * density)
        else:
            self.current_iou = self.iou_threshold
        
        # Appel à la méthode parent
        result = super().update(dets, frame_info)
        
        # Enregistrer les confiances
        for obj in result:
            track_id = int(obj[4])
            # Trouver la détection correspondante
            for det in dets:
                iou = iou_single(obj[:4], det[:4])
                if iou > 0.8:  # Seuil élevé pour être sûr
                    self.track_confidences[track_id].append(det[4])
                    break
        
        return result
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Surcharge pour utiliser le seuil adaptatif.
        """
        if len(trackers) == 0:
            return np.empty((0, 2)), np.arange(len(detections)), np.empty((0,))
        
        iou_matrix = iou_batch(detections[:, :4], trackers[:, :4])
        
        # Algorithme hongrois
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        unmatched_detections = []
        unmatched_trackers = []
        
        # Filtrer avec le seuil adaptatif
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.current_iou:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        # Ajouter les détections non matchées
        for d in range(len(detections)):
            if d not in [m[0] for m in matched_indices]:
                unmatched_detections.append(d)
        
        # Ajouter les trackers non matchés
        for t in range(len(trackers)):
            if t not in [m[1] for m in matched_indices]:
                unmatched_trackers.append(t)
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def get_confidence_stats(self):
        """
        Retourne les statistiques de confiance.
        
        Returns:
            dict: Statistiques par track
        """
        stats = {}
        for track_id, confs in self.track_confidences.items():
            if confs:
                stats[track_id] = {
                    'mean': np.mean(confs),
                    'max': np.max(confs),
                    'min': np.min(confs),
                    'count': len(confs)
                }
        return stats


# =============================================================================
# FONCTION PRINCIPALE DE TEST
# =============================================================================

def test_tracker():
    """Teste le tracker sur des données simulées."""
    print("="*50)
    print("TEST DU MODULE SORT TRACKER")
    print("="*50)
    
    # Créer un tracker
    print("\n1. Initialisation du tracker:")
    tracker = Sort()
    
    # Simuler des détections sur 10 frames
    print("\n2. Test avec données simulées:")
    
    # Simuler 2 objets qui bougent
    np.random.seed(42)
    
    for frame in range(10):
        # Créer des détections simulées
        dets = []
        
        # Objet 1: se déplace vers la droite
        x1 = 100 + frame * 5
        y1 = 100
        dets.append([x1, y1, x1+50, y1+50, 0.9])
        
        # Objet 2: se déplace vers le bas
        x1 = 200
        y1 = 100 + frame * 3
        dets.append([x1, y1, x1+40, y1+40, 0.8])
        
        # Ajouter du bruit (détection aléatoire)
        if frame == 5:
            dets.append([300, 150, 350, 200, 0.7])  # Nouvel objet
        
        dets = np.array(dets)
        
        # Mettre à jour le tracker
        tracks = tracker.update(dets)
        
        print(f"   Frame {frame}: {len(dets)} détections → {len(tracks)} tracks")
    
    # Afficher les statistiques
    print("\n3. Statistiques:")
    stats = tracker.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Tester la version améliorée
    print("\n4. Test de la version améliorée:")
    enhanced = EnhancedSort(adaptive_iou=True)
    
    for frame in range(10):
        # Mêmes détections
        dets = np.array([
            [100 + frame*5, 100, 150 + frame*5, 150, 0.9],
            [200, 100 + frame*3, 240, 140 + frame*3, 0.8]
        ])
        enhanced.update(dets)
    
    print(f"   Version améliorée - objets uniques: {enhanced.get_count()}")
    
    print("\n✅ Test terminé avec succès!")


# =============================================================================
# EXÉCUTION SI LE FICHIER EST LANCÉ DIRECTEMENT
# =============================================================================

if __name__ == "__main__":
    test_tracker()