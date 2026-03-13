#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de visualisation pour la pratique SORT.
Gère l'affichage des détections, des tracks et des graphiques d'analyse.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path
import os
from collections import defaultdict
import pandas as pd

# Importer la configuration
import config

# =============================================================================
# CONFIGURATION DES COULEURS
# =============================================================================

# Couleurs prédéfinies pour les IDs (cycle de 20 couleurs)
COLORS = [
    (255, 0, 0),     # Rouge
    (0, 255, 0),     # Vert
    (0, 0, 255),     # Bleu
    (255, 255, 0),   # Jaune
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 0),     # Bordeaux
    (0, 128, 0),     # Vert foncé
    (0, 0, 128),     # Bleu foncé
    (128, 128, 0),   # Olive
    (128, 0, 128),   # Violet
    (0, 128, 128),   # Teal
    (255, 128, 0),   # Orange
    (255, 0, 128),   # Rose
    (128, 255, 0),   # Vert clair
    (0, 255, 128),   # Vert menthe
    (128, 0, 255),   # Violet clair
    (0, 128, 255),   # Bleu clair
    (255, 128, 128), # Rose clair
    (128, 128, 128)  # Gris
]

def get_color(track_id):
    """
    Retourne une couleur pour un ID de track donné.
    
    Args:
        track_id: ID du track
    
    Returns:
        tuple: Couleur (R, G, B)
    """
    return COLORS[track_id % len(COLORS)]

def get_color_matplotlib(track_id):
    """
    Retourne une couleur au format matplotlib (0-1).
    
    Args:
        track_id: ID du track
    
    Returns:
        tuple: Couleur normalisée (R, G, B) entre 0 et 1
    """
    color = get_color(track_id)
    return (color[0]/255, color[1]/255, color[2]/255)


# =============================================================================
# VISUALISATION DES DÉTECTIONS
# =============================================================================

def draw_detections(image, detections, color=(255, 0, 0), thickness=2, show_confidence=True):
    """
    Dessine les boîtes de détection sur l'image.
    
    Args:
        image: Image numpy (H, W, 3)
        detections: Tableau de détections [N, 5] au format [x1,y1,x2,y2,conf]
        color: Couleur des boîtes (BGR pour OpenCV)
        thickness: Épaisseur du trait
        show_confidence: Afficher la confiance
    
    Returns:
        numpy.ndarray: Image avec les détections
    """
    img_copy = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det[:4].astype(int)
        
        # Dessiner le rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Afficher la confiance
        if show_confidence and len(det) > 4:
            conf = det[4]
            text = f"{conf:.2f}"
            
            # Fond pour le texte
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
            
            # Texte en blanc
            cv2.putText(img_copy, text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def draw_tracks(image, tracks, color_map=None, show_id=True, show_center=True):
    """
    Dessine les tracks avec leurs IDs sur l'image.
    
    Args:
        image: Image numpy (H, W, 3)
        tracks: Tableau de tracks [M, 5] au format [x1,y1,x2,y2,id]
        color_map: Dictionnaire {id: couleur} (optionnel)
        show_id: Afficher l'ID
        show_center: Afficher le centre
    
    Returns:
        tuple: (image_annotee, color_map_mise_a_jour)
    """
    img_copy = image.copy()
    
    if color_map is None:
        color_map = {}
    
    for track in tracks:
        x1, y1, x2, y2, track_id = track.astype(int)
        
        # Obtenir ou créer une couleur pour cet ID
        if track_id not in color_map:
            color_map[track_id] = get_color(track_id)
        color = color_map[track_id]
        
        # Dessiner le rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Afficher l'ID
        if show_id:
            text = f"ID:{track_id}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Fond pour le texte
            cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            
            # Texte en blanc
            cv2.putText(img_copy, text, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dessiner le centre
        if show_center:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img_copy, (cx, cy), 4, color, -1)
            cv2.circle(img_copy, (cx, cy), 6, (255, 255, 255), 1)
    
    return img_copy, color_map


def draw_tracks_with_history(image, tracks, track_history, current_frame, color_map=None):
    """
    Dessine les tracks avec leur historique de trajectoire.
    
    Args:
        image: Image numpy
        tracks: Tracks actuels
        track_history: Dictionnaire {id: [(frame, x, y), ...]}
        current_frame: Numéro de frame actuel
        color_map: Dictionnaire des couleurs
    
    Returns:
        tuple: (image_annotee, color_map)
    """
    img_copy, color_map = draw_tracks(image, tracks, color_map)
    
    # Dessiner les trajectoires récentes
    for track in tracks:
        track_id = int(track[4])
        if track_id in track_history:
            history = track_history[track_id]
            # Garder seulement les 20 dernières positions
            recent = [h for h in history if h[0] > current_frame - 20]
            
            if len(recent) > 1:
                points = []
                for h in recent:
                    x, y = int(h[1]), int(h[2])
                    points.append((x, y))
                
                # Dessiner la ligne de trajectoire
                color = color_map.get(track_id, (255, 255, 255))
                for i in range(1, len(points)):
                    cv2.line(img_copy, points[i-1], points[i], color, 1)
                
                # Marquer les positions précédentes
                for i, (x, y) in enumerate(points[:-1]):
                    alpha = 0.3 + 0.5 * (i / len(points))  # Dégradé
                    cv2.circle(img_copy, (x, y), 2, color, -1)
    
    return img_copy, color_map


# =============================================================================
# VISUALISATION AVEC MATPLOTLIB (POUR LE NOTEBOOK)
# =============================================================================

def plot_detection_examples(image_files, detector, data_loader, num_examples=4, save=True):
    """
    Affiche des exemples de détection dans une grille.
    
    Args:
        image_files: Liste des fichiers images
        detector: Détecteur YOLO
        data_loader: Module de chargement
        num_examples: Nombre d'exemples à afficher
        save: Sauvegarder l'image
    """
    # Sélectionner des images réparties dans la séquence
    indices = np.linspace(0, len(image_files)-1, num_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Charger l'image
        img_file = image_files[idx]
        img = data_loader.load_image(img_file)
        
        # Détecter
        detections = detector.detect(img)
        
        # Afficher
        axes[i].imshow(img)
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            rect = Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(x1, y1-5, f'{conf:.2f}', color='red',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        axes[i].set_title(f"{img_file}\n{len(detections)} objets détectés")
        axes[i].axis('off')
    
    plt.suptitle("Exemples de détection YOLO (classe 4 - букса)", fontsize=14)
    plt.tight_layout()
    
    if save:
        save_path = config.VISUALIZATION_DIR / "detection_examples.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Exemples sauvegardés: {save_path}")
    
    plt.show()


def plot_tracking_statistics(frame_data, track_history, ground_truth=None, save=True):
    """
    Génère les graphiques d'analyse du tracking.
    
    Args:
        frame_data: Liste de dictionnaires avec les données par frame
        track_history: Dictionnaire des historiques de tracks
        ground_truth: Nombre réel d'objets (optionnel)
        save: Sauvegarder l'image
    """
    # Convertir en DataFrame si nécessaire
    if isinstance(frame_data, list):
        df = pd.DataFrame(frame_data)
    else:
        df = frame_data
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Détections et tracks par frame
    ax = axes[0, 0]
    ax.plot(df['frame'], df['detections'], 'b-', label='Détections', alpha=0.7, linewidth=1.5)
    ax.plot(df['frame'], df['tracks'], 'r-', label='Tracks actifs', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Nombre')
    ax.set_title('Détections vs Tracks actifs par frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Évolution du comptage
    ax = axes[0, 1]
    ax.plot(df['frame'], df['unique_count'], 'g-', linewidth=2.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Objets uniques cumulés')
    ax.set_title('Évolution du comptage d\'objets')
    ax.grid(True, alpha=0.3)
    
    if ground_truth:
        ax.axhline(y=ground_truth, color='r', linestyle='--', alpha=0.7, 
                  label=f'Vérité terrain: {ground_truth}')
        ax.legend()
    
    # 3. Distribution de la durée des tracks
    ax = axes[0, 2]
    track_lifetimes = [len(positions) for positions in track_history.values()]
    if track_lifetimes:
        ax.hist(track_lifetimes, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(track_lifetimes), color='red', linestyle='--', 
                  label=f'Moyenne: {np.mean(track_lifetimes):.1f}')
        ax.set_xlabel('Durée de vie (frames)')
        ax.set_ylabel('Nombre de tracks')
        ax.set_title('Distribution de la durée des tracks')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Distribution des détections
    ax = axes[1, 0]
    ax.hist(df['detections'], bins=20, color='salmon', edgecolor='black', alpha=0.7, label='Détections')
    ax.hist(df['tracks'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7, label='Tracks')
    ax.set_xlabel('Nombre')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des détections et tracks')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Évolution du nombre de tracks créés
    ax = axes[1, 1]
    # Compter les nouveaux tracks par frame
    new_tracks_per_frame = defaultdict(int)
    for track_id, positions in track_history.items():
        if positions:
            first_frame = positions[0][0]
            new_tracks_per_frame[first_frame] += 1
    
    frames = sorted(new_tracks_per_frame.keys())
    counts = [new_tracks_per_frame[f] for f in frames]
    
    ax.bar(frames, counts, width=1.0, color='purple', alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Nouveaux tracks')
    ax.set_title('Apparition de nouveaux tracks')
    ax.grid(True, alpha=0.3)
    
    # 6. Ratio de suivi
    ax = axes[1, 2]
    # Éviter la division par zéro
    valid_frames = df['detections'] > 0
    tracking_ratio = np.zeros(len(df))
    tracking_ratio[valid_frames] = df.loc[valid_frames, 'tracks'] / df.loc[valid_frames, 'detections']
    
    ax.plot(df['frame'], tracking_ratio, 'b-', alpha=0.7, linewidth=1.5)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ratio idéal')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Ratio tracks / détections')
    ax.set_title('Qualité du suivi')
    ax.set_ylim(0, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Analyse statistique du tracking SORT", fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save:
        save_path = config.VISUALIZATION_DIR / "tracking_statistics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Graphiques sauvegardés: {save_path}")
    
    plt.show()


def plot_trajectories(track_history, background_image=None, max_tracks=30, save=True):
    """
    Visualise les trajectoires des objets suivis.
    
    Args:
        track_history: Dictionnaire {id: [(frame, x, y), ...]}
        background_image: Image de fond (optionnelle)
        max_tracks: Nombre maximum de tracks à afficher
        save: Sauvegarder l'image
    """
    plt.figure(figsize=(16, 10))
    
    # Afficher le fond si fourni
    if background_image is not None:
        plt.imshow(background_image, alpha=0.3)
    
    # Sélectionner les tracks les plus longs
    tracks_with_length = [(tid, len(positions)) for tid, positions in track_history.items()]
    tracks_with_length.sort(key=lambda x: x[1], reverse=True)
    top_tracks = [tid for tid, _ in tracks_with_length[:max_tracks]]
    
    # Dessiner les trajectoires
    for track_id in top_tracks:
        positions = track_history[track_id]
        if len(positions) < 5:  # Ignorer les tracks trop courts
            continue
        
        frames, xs, ys = zip(*positions)
        
        # Obtenir une couleur
        color = get_color_matplotlib(track_id)
        
        # Tracer la trajectoire
        plt.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7, label=f'ID:{track_id}')
        
        # Marquer le début et la fin
        plt.scatter(xs[0], ys[0], c=[color], s=100, marker='o', 
                   edgecolors='black', linewidths=2, zorder=5, label='_nolegend_')
        plt.scatter(xs[-1], ys[-1], c=[color], s=100, marker='s', 
                   edgecolors='black', linewidths=2, zorder=5, label='_nolegend_')
        
        # Ajouter le numéro de frame au début
        plt.annotate(f'{frames[0]}', (xs[0], ys[0]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.title(f'Trajectoires des objets (échantillon de {len(top_tracks)} tracks)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        save_path = config.VISUALIZATION_DIR / "trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Trajectoires sauvegardées: {save_path}")
    
    plt.show()


def plot_tracking_summary(tracker, frame_data, ground_truth=None):
    """
    Affiche un résumé complet du tracking.
    
    Args:
        tracker: Instance du tracker SORT
        frame_data: Données par frame
        ground_truth: Vérité terrain (optionnel)
    """
    df = pd.DataFrame(frame_data)
    stats = tracker.get_stats()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphique en barres des statistiques
    ax = axes[0]
    metrics = ['Objets uniques', 'Tracks créés', 'Frames']
    values = [stats['unique_objects'], stats['total_tracks_created'], stats['frames_processed']]
    
    bars = ax.bar(metrics, values, color=['green', 'blue', 'orange'])
    ax.set_title('Résumé global')
    ax.set_ylabel('Nombre')
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(val), ha='center', va='bottom')
    
    # Comparaison avec vérité terrain
    ax = axes[1]
    if ground_truth:
        comparison = ['Notre comptage', 'Vérité terrain']
        values_comp = [stats['unique_objects'], ground_truth]
        colors_comp = ['green', 'red']
        
        bars = ax.bar(comparison, values_comp, color=colors_comp)
        ax.set_title('Comparaison avec vérité terrain')
        ax.set_ylabel('Nombre')
        
        # Ajouter l'erreur
        error = abs(stats['unique_objects'] - ground_truth)
        accuracy = (1 - error/ground_truth) * 100 if ground_truth > 0 else 0
        
        for bar, val in zip(bars, values_comp):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(val), ha='center', va='bottom')
        
        ax.text(0.5, 0.9, f'Erreur: {error} | Précision: {accuracy:.1f}%',
               transform=ax.transAxes, ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# CRÉATION DE VIDÉO ET ANIMATIONS
# =============================================================================

def save_visualization_frame(image, tracks, frame_idx, output_dir=None, color_map=None):
    """
    Sauvegarde une frame avec les tracks pour créer une vidéo.
    
    Args:
        image: Image numpy
        tracks: Tracks à dessiner
        frame_idx: Index de la frame
        output_dir: Dossier de sortie
        color_map: Dictionnaire des couleurs
    
    Returns:
        dict: Color_map mis à jour
    """
    if output_dir is None:
        output_dir = config.FRAMES_DIR
    
    output_dir.mkdir(exist_ok=True)
    
    # Dessiner les tracks
    img_with_tracks, color_map = draw_tracks(image, tracks, color_map)
    
    # Ajouter le numéro de frame
    cv2.putText(img_with_tracks, f"Frame: {frame_idx}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Sauvegarder
    output_path = output_dir / f"frame_{frame_idx:04d}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(img_with_tracks, cv2.COLOR_RGB2BGR))
    
    return color_map


def create_video_from_frames(frames_dir=None, output_path=None, fps=10):
    """
    Crée une vidéo à partir des frames sauvegardées.
    
    Args:
        frames_dir: Dossier contenant les frames
        output_path: Chemin de sortie de la vidéo
        fps: Images par seconde
    """
    if frames_dir is None:
        frames_dir = config.FRAMES_DIR
    
    if output_path is None:
        output_path = config.OUTPUT_DIR / "tracking_video.mp4"
    
    # Récupérer toutes les frames
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    
    if not frame_files:
        print("⚠️ Aucune frame trouvée")
        return
    
    # Lire la première frame pour les dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape
    
    # Initialiser le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Ajouter toutes les frames
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        video_writer.write(frame)
    
    video_writer.release()
    print(f"✓ Vidéo créée: {output_path}")


# =============================================================================
# FONCTIONS D'AIDE À L'ANALYSE
# =============================================================================

def plot_confidence_analysis(tracker_enhanced, save=True):
    """
    Analyse les confiances des tracks (pour la version améliorée).
    
    Args:
        tracker_enhanced: Instance de EnhancedSort
        save: Sauvegarder l'image
    """
    if not hasattr(tracker_enhanced, 'get_confidence_stats'):
        print("❌ Cette fonction nécessite un tracker amélioré")
        return
    
    conf_stats = tracker_enhanced.get_confidence_stats()
    
    if not conf_stats:
        print("⚠️ Aucune statistique de confiance disponible")
        return
    
    # Préparer les données
    track_ids = list(conf_stats.keys())
    means = [conf_stats[tid]['mean'] for tid in track_ids]
    maxs = [conf_stats[tid]['max'] for tid in track_ids]
    mins = [conf_stats[tid]['min'] for tid in track_ids]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogramme des confiances moyennes
    ax = axes[0, 0]
    ax.hist(means, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confiance moyenne')
    ax.set_ylabel('Nombre de tracks')
    ax.set_title('Distribution des confiances moyennes par track')
    ax.grid(True, alpha=0.3)
    
    # Box plot des confiances
    ax = axes[0, 1]
    # Prendre un échantillon de tracks
    sample_ids = track_ids[:min(20, len(track_ids))]
    sample_data = [list(tracker_enhanced.track_confidences[tid]) for tid in sample_ids]
    
    bp = ax.boxplot(sample_data, patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_alpha(0.6)
    ax.set_xlabel('Track ID')
    ax.set_ylabel('Confiance')
    ax.set_title('Distribution des confiances par track (échantillon)')
    ax.grid(True, alpha=0.3)
    
    # Évolution de la confiance
    ax = axes[1, 0]
    for tid in sample_ids[:5]:  # 5 premiers tracks
        confs = tracker_enhanced.track_confidences[tid]
        ax.plot(confs, 'o-', label=f'ID:{tid}', alpha=0.7, markersize=4)
    ax.set_xlabel('Détection #')
    ax.set_ylabel('Confiance')
    ax.set_title('Évolution de la confiance (échantillon)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistiques globales
    ax = axes[1, 1]
    ax.axis('off')
    
    all_confs = []
    for tid in track_ids:
        all_confs.extend(tracker_enhanced.track_confidences[tid])
    
    text = f"""
    STATISTIQUES GLOBALES DE CONFIANCE
    
    Nombre total de tracks: {len(track_ids)}
    Nombre total de détections: {len(all_confs)}
    
    Confiance moyenne globale: {np.mean(all_confs):.3f}
    Écart-type: {np.std(all_confs):.3f}
    Confiance minimale: {np.min(all_confs):.3f}
    Confiance maximale: {np.max(all_confs):.3f}
    
    Tracks avec confiance > 0.8: {sum(m > 0.8 for m in means)}
    Tracks avec confiance < 0.3: {sum(m < 0.3 for m in means)}
    """
    
    ax.text(0.1, 0.5, text, fontsize=12, va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle("Analyse détaillée des confiances", fontsize=16)
    plt.tight_layout()
    
    if save:
        save_path = config.VISUALIZATION_DIR / "confidence_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Analyse des confiances sauvegardée: {save_path}")
    
    plt.show()


# =============================================================================
# FONCTION PRINCIPALE DE TEST
# =============================================================================

def test_visualization():
    """Teste les fonctions de visualisation avec des données simulées."""
    print("="*50)
    print("TEST DU MODULE VISUALIZATION")
    print("="*50)
    
    # Créer une image de test
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gris
    
    # Créer des détections simulées
    detections = np.array([
        [100, 100, 150, 150, 0.95],
        [200, 200, 280, 280, 0.85],
        [400, 300, 500, 400, 0.75]
    ])
    
    print("\n1. Test draw_detections:")
    img_det = draw_detections(test_image, detections)
    print("   ✓ Détections dessinées")
    
    # Créer des tracks simulés
    tracks = np.array([
        [100, 100, 150, 150, 1],
        [200, 200, 280, 280, 2],
        [400, 300, 500, 400, 3]
    ])
    
    print("\n2. Test draw_tracks:")
    img_trk, color_map = draw_tracks(test_image, tracks)
    print("   ✓ Tracks dessinés")
    
    # Tester la sauvegarde
    print("\n3. Test save_visualization_frame:")
    save_visualization_frame(test_image, tracks, 0)
    print(f"   ✓ Frame sauvegardée dans {config.FRAMES_DIR}")
    
    print("\n✅ Tests terminés avec succès!")


# =============================================================================
# EXÉCUTION SI LE FICHIER EST LANCÉ DIRECTEMENT
# =============================================================================

if __name__ == "__main__":
    test_visualization()