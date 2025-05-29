import cv2
import urllib.request
import numpy as np
import threading
import time
import socket
import json
from collections import deque
import struct

# Configuration
URL1 = "http://192.168.1.100:4747/video"
URL2 = "http://192.168.1.106:4747/video"

# Configuration TCP pour synchronisation
TCP_HOST = '192.168.1.100'  # IP du téléphone maître
TCP_PORT = 12345
SYNC_BUFFER_SIZE = 30  # Nombre de frames à garder en buffer
MAX_SYNC_DELAY = 0.1   # Délai max acceptable entre frames (100ms)

# Variables globales
frame_buffer1 = deque(maxlen=SYNC_BUFFER_SIZE)
frame_buffer2 = deque(maxlen=SYNC_BUFFER_SIZE)
lock1 = threading.Lock()
lock2 = threading.Lock()
sync_lock = threading.Lock()

# Variables de synchronisation
sync_offset = 0.0  # Décalage temporel entre les deux flux
is_master = True   # Premier téléphone = maître
sync_server = None
sync_client = None

class FrameData:
    def __init__(self, frame, timestamp, frame_id):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_id = frame_id

def setup_tcp_sync():
    """Configure la synchronisation TCP entre les appareils"""
    global sync_server, sync_client, is_master
    
    try:
        # Tenter de créer un serveur (appareil maître)
        sync_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sync_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sync_server.bind((TCP_HOST, TCP_PORT))
        sync_server.listen(1)
        sync_server.settimeout(2.0)
        
        print(f"Serveur de sync démarré sur {TCP_HOST}:{TCP_PORT}")
        print("En attente de connexion de l'appareil esclave...")
        
        conn, addr = sync_server.accept()
        print(f"Appareil esclave connecté: {addr}")
        is_master = True
        return conn
        
    except socket.timeout:
        print("Aucun autre appareil trouvé, mode esclave activé")
        sync_server.close()
        
        # Se connecter comme client (appareil esclave)
        sync_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sync_client.connect((TCP_HOST, TCP_PORT))
        is_master = False
        return sync_client
        
    except Exception as e:
        print(f"Erreur de configuration TCP: {e}")
        is_master = True
        return None

def sync_timestamps(connection):
    """Synchronise les timestamps entre les appareils"""
    global sync_offset
    
    if not connection:
        return
        
    try:
        if is_master:
            # Envoyer timestamp de référence
            ref_time = time.time()
            sync_data = {
                'type': 'sync',
                'timestamp': ref_time,
                'frame_id': int(ref_time * 1000) % 10000
            }
            message = json.dumps(sync_data).encode() + b'\n'
            connection.send(message)
            
        else:
            # Recevoir et calculer l'offset
            data = connection.recv(1024).decode().strip()
            if data:
                sync_data = json.loads(data)
                if sync_data['type'] == 'sync':
                    current_time = time.time()
                    sync_offset = sync_data['timestamp'] - current_time
                    print(f"Offset de synchronisation calculé: {sync_offset:.3f}s")
                    
    except Exception as e:
        print(f"Erreur de synchronisation: {e}")

def get_synchronized_timestamp():
    """Retourne un timestamp synchronisé"""
    return time.time() + sync_offset

def read_stream_synchronized(url, buffer, lock, stream_id):
    """Lit un flux vidéo avec timestamps synchronisés"""
    stream = urllib.request.urlopen(url)
    bytes_data = b''
    frame_counter = 0

    while True:
        try:
            bytes_data += stream.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    timestamp = get_synchronized_timestamp()
                    frame_data = FrameData(frame, timestamp, frame_counter)
                    
                    with lock:
                        buffer.append(frame_data)
                    
                    frame_counter += 1
                    
        except Exception as e:
            print(f"Erreur dans le flux {url}: {e}")
            break

def find_synchronized_frames():
    """Trouve les frames les mieux synchronisées entre les deux flux"""
    with lock1:
        buffer1_copy = list(frame_buffer1)
    with lock2:
        buffer2_copy = list(frame_buffer2)
    
    if not buffer1_copy or not buffer2_copy:
        return None, None
    
    best_frame1 = None
    best_frame2 = None
    min_time_diff = float('inf')
    
    # Rechercher les frames avec le plus petit écart temporel
    for frame1 in buffer1_copy:
        for frame2 in buffer2_copy:
            time_diff = abs(frame1.timestamp - frame2.timestamp)
            if time_diff < min_time_diff and time_diff < MAX_SYNC_DELAY:
                min_time_diff = time_diff
                best_frame1 = frame1
                best_frame2 = frame2
    
    return best_frame1, best_frame2, min_time_diff if best_frame1 else None

def sync_manager():
    """Gestionnaire de synchronisation TCP en arrière-plan"""
    connection = setup_tcp_sync()
    
    while True:
        try:
            sync_timestamps(connection)
            time.sleep(1.0)  # Synchroniser chaque seconde
        except Exception as e:
            print(f"Erreur dans le gestionnaire de sync: {e}")
            break
    
    if connection:
        connection.close()

def main():
    print("=== SYSTÈME DE SYNCHRONISATION DUAL-CAMÉRA ===")
    print("Démarrage de la synchronisation TCP...")
    
    # Lancer le gestionnaire de synchronisation
    sync_thread = threading.Thread(target=sync_manager, daemon=True)
    sync_thread.start()
    
    # Attendre un peu pour établir la synchronisation
    time.sleep(3)
    
    # Lancer les threads de lecture des flux
    thread1 = threading.Thread(target=read_stream_synchronized, 
                              args=(URL1, frame_buffer1, lock1, "stream1"), daemon=True)
    thread2 = threading.Thread(target=read_stream_synchronized, 
                              args=(URL2, frame_buffer2, lock2, "stream2"), daemon=True)
    
    thread1.start()
    thread2.start()
    
    print("Flux démarrés. Appuyez sur 'q' pour quitter.")
    print("Appuyez sur 's' pour afficher les statistiques de sync.")
    
    # Variables de statistiques
    sync_stats = {'good_sync': 0, 'poor_sync': 0, 'no_sync': 0}
    
    try:
        while True:
            result = find_synchronized_frames()
            
            if result and len(result) == 3:
                frame1_data, frame2_data, time_diff = result
                
                if frame1_data and frame2_data:
                    f1 = frame1_data.frame.copy()
                    f2 = frame2_data.frame.copy()
                    
                    # Statistiques de synchronisation
                    if time_diff < 0.033:  # < 33ms (bon)
                        sync_stats['good_sync'] += 1
                        sync_color = (0, 255, 0)  # Vert
                        sync_status = "SYNC OK"
                    elif time_diff < 0.1:  # < 100ms (acceptable)
                        sync_stats['poor_sync'] += 1
                        sync_color = (0, 255, 255)  # Jaune
                        sync_status = "SYNC MOYEN"
                    else:
                        sync_stats['no_sync'] += 1
                        sync_color = (0, 0, 255)  # Rouge
                        sync_status = "DÉSYNCHRONISÉ"
                    
                    # Redimensionner si nécessaire
                    if f1.shape != f2.shape:
                        f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
                    
                    # Ajouter informations de synchronisation
                    cv2.putText(f1, f"T1: {frame1_data.timestamp:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(f2, f"T2: {frame2_data.timestamp:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Combiner les frames
                    combined = cv2.hconcat([f1, f2])
                    
                    # Ajouter statut de synchronisation
                    cv2.putText(combined, f"{sync_status} - Diff: {time_diff*1000:.1f}ms", 
                               (10, combined.shape[0] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, sync_color, 2)
                    
                    cv2.putText(combined, f"Offset: {sync_offset*1000:.1f}ms", 
                               (10, combined.shape[0] - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow("Flux synchronisés TCP (Appareil 1 | Appareil 2)", combined)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                total = sum(sync_stats.values())
                if total > 0:
                    print(f"\n=== STATISTIQUES DE SYNCHRONISATION ===")
                    print(f"Bonne sync (<33ms): {sync_stats['good_sync']} ({sync_stats['good_sync']/total*100:.1f}%)")
                    print(f"Sync moyenne (<100ms): {sync_stats['poor_sync']} ({sync_stats['poor_sync']/total*100:.1f}%)")
                    print(f"Désynchronisé (>100ms): {sync_stats['no_sync']} ({sync_stats['no_sync']/total*100:.1f}%)")
                    print(f"Offset actuel: {sync_offset*1000:.1f}ms")
    
    except KeyboardInterrupt:
        print("\nArrêt manuel détecté")
    
    finally:
        cv2.destroyAllWindows()
        print("Fermeture des connexions...")
        
        # Fermer les connexions TCP
        if sync_server:
            sync_server.close()
        if sync_client:
            sync_client.close()
        
        print("Nettoyage terminé.")

if __name__ == "__main__":
    main()