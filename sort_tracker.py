#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module SORT simplifié pour Google Colab
"""

import numpy as np
from collections import defaultdict

class SimpleTracker:
    count = 0
    
    def __init__(self, bbox):
        self.bbox = [float(x) for x in bbox]
        self.time_since_update = 0
        self.id = SimpleTracker.count
        SimpleTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.start_frame = None
    
    def update(self, bbox, frame_num=None):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        if frame_num is not None and self.start_frame is None:
            self.start_frame = frame_num
        self.bbox = [float(x) for x in bbox]
    
    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.bbox
    
    def get_state(self):
        return np.array(self.bbox)
    
    def get_center(self):
        return ((self.bbox[0] + self.bbox[2])/2, (self.bbox[1] + self.bbox[3])/2)


def compute_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.unique_objects = set()
        self.track_history = defaultdict(list)
    
    def update(self, detections):
        """
        Met à jour les trackers avec les nouvelles détections
        
        Args:
            detections: numpy array de forme (N, 5) ou liste vide
        """
        self.frame_count += 1
        
        # Prédire pour tous les trackers
        for trk in self.trackers:
            trk.predict()
        
        # Si pas de détections
        if len(detections) == 0:
            self._cleanup_old_trackers()
            return np.empty((0, 5))
        
        # Convertir en liste pour faciliter le traitement
        if isinstance(detections, np.ndarray):
            dets_list = [detections[i] for i in range(len(detections))]
        else:
            dets_list = detections
        
        # Matcher les détections avec les trackers
        matches, unmatched_dets, unmatched_trks = self._match(dets_list)
        
        # Mettre à jour les trackers matchés
        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(dets_list[det_idx][:4], self.frame_count)
            center = self.trackers[trk_idx].get_center()
            self.track_history[self.trackers[trk_idx].id].append(
                (self.frame_count, center[0], center[1])
            )
            self.unique_objects.add(self.trackers[trk_idx].id)
        
        # Créer de nouveaux trackers pour les détections non matchées
        for det_idx in unmatched_dets:
            trk = SimpleTracker(dets_list[det_idx][:4])
            trk.start_frame = self.frame_count
            self.trackers.append(trk)
            center = trk.get_center()
            self.track_history[trk.id] = [(self.frame_count, center[0], center[1])]
            self.unique_objects.add(trk.id)
        
        # Nettoyer les vieux trackers
        self._cleanup_old_trackers()
        
        # Préparer les résultats
        results = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    state = trk.get_state()
                    results.append(np.concatenate([state, [trk.id + 1]]))
        
        return np.array(results) if results else np.empty((0, 5))
    
    def _match(self, detections):
        """Match les détections avec les trackers"""
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.trackers)))
        
        # Calculer toutes les IoU
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(self.trackers):
                iou_matrix[i, j] = compute_iou(det[:4], trk.get_state())
        
        # Trouver les meilleurs matches
        while len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            best_iou = self.iou_threshold
            best_pair = None
            
            for i in unmatched_dets:
                for j in unmatched_trks:
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_pair = (i, j)
            
            if best_pair is None:
                break
            
            i, j = best_pair
            matches.append((i, j))
            unmatched_dets.remove(i)
            unmatched_trks.remove(j)
        
        return matches, unmatched_dets, unmatched_trks
    
    def _cleanup_old_trackers(self):
        """Supprime les trackers trop âgés"""
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
    
    def get_count(self):
        return len(self.unique_objects)
    
    def get_stats(self):
        track_lifetimes = []
        for trk_id, history in self.track_history.items():
            if len(history) > 0:
                frames = [h[0] for h in history]
                lifetime = max(frames) - min(frames) + 1
                track_lifetimes.append(lifetime)
        
        return {
            'unique_objects': len(self.unique_objects),
            'total_tracks_created': SimpleTracker.count,
            'active_tracks': len(self.trackers),
            'frames_processed': self.frame_count,
            'avg_track_lifetime': np.mean(track_lifetimes) if track_lifetimes else 0
        }
