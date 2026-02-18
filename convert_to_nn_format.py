"""
Script para convertir datos de simulación EDM4HEP a formato simplificado para entrenamiento de redes neuronales.

Genera dos conjuntos de datos por evento:
1. Hits: colección de todos los hits de detectores (Inner tracker, ECAL, HCAL, Muon tracker)
2. Particles: colección de partículas asociadas a los hits

Uso:
    python convert_to_nn_format.py input_file.root output_file.npz [CLIC]
    
Ejemplo:
    python convert_to_nn_format.py /path/to/out_reco_edm4hep.root data_for_training.npz True
"""

import sys
import math
import numpy as np
from podio import root_io
import edm4hep
import logging
import time
from datetime import datetime, timedelta

# Constantes globales
C_LIGHT = 2.99792458e8  # m/s
BZ_CLIC = 4.0  # Tesla
BZ_CLD = 2.0   # Tesla
MCHP = 0.139570  # GeV (masa del pión cargado)

# Tipos de detectores
DETECTOR_TYPES = {
    'INNER_TRACKER': 0,
    'ECAL': 1,
    'HCAL': 2,
    'MUON_TRACKER': 3
}


def setup_logging(output_file):
    """Configura el sistema de logging con archivo y consola."""
    log_filename = output_file.replace('.npz', '_conversion.log')
    
    # Configurar formato
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Crear handlers con flush inmediato
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # Configurar logging root
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return log_filename


def omega_to_pt(omega, Bz):
    """Convierte omega de track state a pt transversal."""
    a = C_LIGHT * 1e3 * 1e-15
    return a * Bz / abs(omega)


def track_momentum(trackstate, is_clic=True):
    """
    Calcula el momento completo de una traza a partir del track state.
    
    Returns:
        p, theta, phi, energy, px, py, pz
    """
    Bz = BZ_CLIC if is_clic else BZ_CLD
    pt = omega_to_pt(trackstate.omega, Bz)
    phi = trackstate.phi
    pz = trackstate.tanLambda * pt
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    p = math.sqrt(px * px + py * py + pz * pz)
    energy = math.sqrt(p * p + MCHP * MCHP)
    theta = math.acos(pz / p) if p > 0 else 0
    
    return p, theta, phi, energy, px, py, pz


def get_genparticle_parents(idx, mcparts):
    """Obtiene los índices de las partículas madre."""
    p = mcparts[idx]
    parents = p.getParents()
    parent_positions = []
    for parent in parents:
        parent_positions.append(parent.getObjectID().index)
    return parent_positions


def find_mother_particle(j, gen_part_coll):
    """
    Encuentra la partícula madre original siguiendo la cadena de padres.
    Útil para rastrear el origen de partículas secundarias.
    """
    parent_p = j
    pp_old = j  # Inicializar para evitar error si el bucle no entra
    counter = 0
    max_iterations = 100  # Prevenir loops infinitos
    
    while counter < max_iterations:
        if isinstance(parent_p, list):
            if len(parent_p) > 0:
                parent_p = parent_p[0]
            else:
                break
        
        parent_p_r = get_genparticle_parents(parent_p, gen_part_coll)
        
        if len(parent_p_r) == 0:
            break
            
        pp_old = parent_p
        parent_p = parent_p_r
        counter += 1
        
        # Detectar loops circulares
        if parent_p == pp_old or parent_p == j:
            print(f"Warning: Circular parent structure detected for particle {j}")
            break
        
    return pp_old if isinstance(pp_old, int) else j


def flush_all():
    """Fuerza el flush de todos los streams de salida."""
    sys.stdout.flush()
    sys.stderr.flush()
    for handler in logging.getLogger().handlers:
        handler.flush()


def estimate_time_remaining(start_time, current, total):
    """Estima el tiempo restante basándose en el progreso actual."""
    if current == 0:
        return "calculando..."
    
    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0
    
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    remaining_str = str(timedelta(seconds=int(remaining)))
    
    return f"Transcurrido: {elapsed_str}, Restante: {remaining_str}, Tasa: {rate:.2f} ev/s"


def find_gen_link(hit_idx, collection_id, truth_links, genpart_map):
    """
    Encuentra las partículas generadas asociadas a un hit.
    
    Returns:
        gen_indices: lista de índices de partículas asociadas
        gen_weights: lista de pesos (fracción de energía)
    """
    gen_positions = []
    gen_weights = []
    
    for link in truth_links:
        rec = link.getRec()
        object_id = rec.getObjectID()
        index = object_id.index
        collectionID = object_id.collectionID
        
        if index == hit_idx and collectionID == collection_id:
            gen_positions.append(link.getSim().getObjectID().index)
            weight = link.getWeight()
            gen_weights.append(weight)
    
    # Mapear a las partículas guardadas
    indices = []
    for pos in gen_positions:
        if pos in genpart_map:
            indices.append(genpart_map[pos])
    
    return indices, gen_weights


def process_gen_particles(event, debug=False):
    """
    Procesa las partículas generadas (MC truth).
    
    Returns:
        particles_data: array con información de partículas
        genpart_map: diccionario de mapeo (índice original MCParticles -> índice en particles_list)
    """
    gen_part_coll = event.get("MCParticles")
    
    particles_list = []
    genpart_map = {}  # índice original MCParticles -> posición en particles_list
    
    for j, part in enumerate(gen_part_coll):
        momentum = part.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        
        if p < 1e-10:  # Omitir partículas con momento muy bajo
            continue
        
        theta = math.acos(momentum.z / p) if p > 0 else 0
        phi = math.atan2(momentum.y, momentum.x)
        
        # El índice en particles_list (el que se usará para asociaciones)
        stored_idx = len(particles_list)
        
        particle_data = {
            'pid': part.getPDG(),
            'charge': part.getCharge(),
            'px': momentum.x,
            'py': momentum.y,
            'pz': momentum.z,
            'p': p,
            'energy': part.getEnergy(),
            'mass': part.getMass(),
            'theta': theta,
            'phi': phi,
            'vx': part.getVertex().x,
            'vy': part.getVertex().y,
            'vz': part.getVertex().z,
            'gen_status': part.getGeneratorStatus(),
            'decayed_in_tracker': int(part.isDecayedInTracker()),
            'decayed_in_calo': int(part.isDecayedInCalorimeter()),
            'mc_index': j  # Guardar también el índice original de MCParticles
        }
        
        genpart_map[j] = stored_idx
        particles_list.append(particle_data)
        
        if debug and stored_idx < 30:
            # Mostrar AMBOS índices: el original (MCParticles) y el guardado (particles_list)
            print(f"Gen Particle [stored_idx={stored_idx}, mc_idx={j}]: PID={particle_data['pid']}, "
                  f"P={p:.3f} GeV, Charge={particle_data['charge']}")
    
    return particles_list, genpart_map


def process_tracks(event, genpart_map, particles_list, is_clic=False, debug=False):
    """
    Procesa las trazas del Inner Tracker.
    
    Returns:
        hits_list: lista de diccionarios con información de hits
    """
    tracks_coll = event.get("SiTracks_Refitted")
    truth_links = event.get("SiTracksMCTruthLink")
    
    hits_list = []
    
    for idx, track in enumerate(tracks_coll):
        track_states = track.getTrackStates()
        
        if len(track_states) == 0:
            continue
        
        # Usar el primer track state (en el vértice) para calcular el momento
        ts_vertex = track_states[0]
        
        # Calcular momento desde el track state en el vértice
        p, theta, phi, energy, px, py, pz = track_momentum(ts_vertex, is_clic)
        
        # Posición: intentar usar el track state en el calorímetro (índice 3) si existe
        # Los track states típicos son:
        #   0: en el vértice (IP)
        #   1: primer hit
        #   2: último hit
        #   3: en el calorímetro (entrada ECAL)
        if len(track_states) >= 4:
            ts_calo = track_states[3]
            ref_point = ts_calo.referencePoint
            x, y, z = ref_point.x, ref_point.y, ref_point.z
        else:
            # Si no hay track state en calorímetro, usar el del vértice
            ref_point = ts_vertex.referencePoint
            x, y, z = ref_point.x, ref_point.y, ref_point.z
        
        # Para gen links: usar el índice del loop (idx) y el collectionID del track
        # Los MCTruthLinks usan índices secuenciales, no objectID.index
        track_collection_id = track.getObjectID().collectionID
        
        # Encontrar partículas generadas asociadas
        gen_indices, gen_weights = find_gen_link(
            idx, track_collection_id, truth_links, genpart_map
        )
        
        hit_data = {
            'detector_type': DETECTOR_TYPES['INNER_TRACKER'],
            'x': x,
            'y': y,
            'z': z,
            'energy': energy,
            'p': p,
            'px': px,
            'py': py,
            'pz': pz,
            'theta': theta,
            'phi': phi,
            'gen_particle_indices': gen_indices,
            'gen_weights': gen_weights
        }
        
        hits_list.append(hit_data)
        
        if debug and idx < 5:
            # Mostrar información detallada de las partículas asociadas
            particle_info = []
            for gi in gen_indices:
                if gi >= 0 and gi < len(particles_list):
                    p_data = particles_list[gi]
                    particle_info.append(f"idx={gi}:PID={p_data['pid']}")
            print(f"Track {idx}: P={p:.3f} GeV, theta={theta:.3f}, phi={phi:.3f}, "
                  f"pos=({x:.1f}, {y:.1f}, {z:.1f}), gen_links={particle_info}")
    
    return hits_list


def process_calo_hits(event, genpart_map, gen_part_coll, is_clic=True, debug=False, log_progress=False):
    """
    Procesa los hits de calorímetros (ECAL, HCAL).
    
    Returns:
        hits_list: lista de diccionarios con información de hits
    """
    hits_list = []
    
    # IMPORTANTE: Usar las colecciones reconstruidas (ECALBarrel, HCALBarrel, etc.)
    # NO las colecciones de simulación (ECalBarrelCollection, etc.)
    # Los CalohitMCTruthLink apuntan a las colecciones reconstruidas
    
    # Procesar ECAL - colecciones reconstruidas
    ecal_collections = [
        ("ECALBarrel", DETECTOR_TYPES['ECAL']),
        ("ECALEndcap", DETECTOR_TYPES['ECAL']),
    ]
    
    # Procesar HCAL - colecciones reconstruidas
    hcal_collections = [
        ("HCALBarrel", DETECTOR_TYPES['HCAL']),
        ("HCALEndcap", DETECTOR_TYPES['HCAL']),
        ("HCALOther", DETECTOR_TYPES['HCAL']),
    ]
    
    all_collections = ecal_collections + hcal_collections
    truth_links = event.get("CalohitMCTruthLink")
    
    # Debug: mostrar los primeros truth links para verificar IDs
    if debug:
        logging.info(f"  Truth links count: {len(truth_links)}")
        for i, link in enumerate(truth_links):
            if i < 5:
                rec_id = link.getRec().getObjectID()
                logging.info(f"    TruthLink {i}: rec.index={rec_id.index}, rec.collectionID={rec_id.collectionID}")
    
    total_hits_processed = 0
    for coll_idx, (coll_name, detector_type) in enumerate(all_collections):
        try:
            calo_coll = event.get(coll_name)
        except:
            if debug:
                logging.debug(f"Warning: Collection {coll_name} not found")
            continue
        
        if len(calo_coll) == 0:
            continue
        
        coll_size = len(calo_coll)
        if log_progress:
            logging.info(f"  Processing {coll_name}: {coll_size} hits")
        
        for idx, hit in enumerate(calo_coll):
            total_hits_processed += 1
            position = hit.getPosition()
            x, y, z = position.x, position.y, position.z
            energy = hit.getEnergy()
            
            # Calcular coordenadas esféricas
            r = math.sqrt(x**2 + y**2 + z**2)
            if r > 0:
                theta = math.acos(z / r)
                phi = math.atan2(y, x)
            else:
                theta = 0
                phi = 0
            
            # Para gen links: usar idx del enumerate (igual que tree_tools.py)
            # y el collectionID del hit
            hit_collection_id = hit.getObjectID().collectionID
            
            # Debug: mostrar los primeros hits para verificar IDs
            if debug and idx < 3:
                logging.info(f"    Hit {idx}: objectID.index={hit.getObjectID().index}, collectionID={hit_collection_id}")
            
            # Encontrar partículas generadas asociadas
            gen_indices, gen_weights = find_gen_link(
                idx, hit_collection_id, truth_links, genpart_map
            )
            
            # Si no hay links directos, intentar encontrar la partícula madre
            if len(gen_indices) == 0 and len(gen_weights) > 0:
                # Esto puede indicar una partícula secundaria
                pass
            
            hit_data = {
                'detector_type': detector_type,
                'x': x,
                'y': y,
                'z': z,
                'energy': energy,
                'p': 0.0,  # Los hits de calo no tienen momento directo
                'px': 0.0,
                'py': 0.0,
                'pz': 0.0,
                'theta': theta,
                'phi': phi,
                'gen_particle_indices': gen_indices,
                'gen_weights': gen_weights
            }
            
            hits_list.append(hit_data)
    
    if log_progress or debug:
        logging.info(f"  Total calorimeter hits processed: {len(hits_list)}")
    
    return hits_list


def process_muon_hits(event, genpart_map, debug=False):
    """
    Procesa los hits del sistema de muones.
    
    Returns:
        hits_list: lista de diccionarios con información de hits
    """
    hits_list = []
    
    try:
        # IMPORTANTE: Usar "MUON" que es la colección reconstruida con truth links
        # NO "YokeBarrelCollection" que es la colección de simulación
        muon_coll = event.get("MUON")
        truth_links = event.get("CalohitMCTruthLink")  # Mismos truth links que calo
        
        for idx, hit in enumerate(muon_coll):
            position = hit.getPosition()
            x, y, z = position.x, position.y, position.z
            energy = hit.getEnergy()
            
            r = math.sqrt(x**2 + y**2 + z**2)
            if r > 0:
                theta = math.acos(z / r)
                phi = math.atan2(y, x)
            else:
                theta = 0
                phi = 0
            
            # Para gen links: usar idx del enumerate (igual que tree_tools.py)
            # y el collectionID del hit
            hit_collection_id = hit.getObjectID().collectionID
            
            gen_indices, gen_weights = find_gen_link(
                idx, hit_collection_id, truth_links, genpart_map
            )
            
            hit_data = {
                'detector_type': DETECTOR_TYPES['MUON_TRACKER'],
                'x': x,
                'y': y,
                'z': z,
                'energy': energy,
                'p': 0.0,
                'px': 0.0,
                'py': 0.0,
                'pz': 0.0,
                'theta': theta,
                'phi': phi,
                'gen_particle_indices': gen_indices,
                'gen_weights': gen_weights
            }
            
            hits_list.append(hit_data)
    except:
        if debug:
            print("Warning: Muon collection not found or error processing")
    
    return hits_list


def convert_to_arrays(particles_list, hits_list):
    """
    Convierte las listas de diccionarios a arrays de NumPy estructurados.
    
    Returns:
        particles_array: array de partículas
        hits_array: array de hits
        hit_to_particle: lista de listas con índices de partículas por hit
        hit_weights: lista de listas con pesos por hit
    """
    # Crear array de partículas
    n_particles = len(particles_list)
    particles_array = np.zeros(n_particles, dtype=[
        ('pid', 'i4'),
        ('charge', 'f4'),
        ('px', 'f4'),
        ('py', 'f4'),
        ('pz', 'f4'),
        ('p', 'f4'),
        ('energy', 'f4'),
        ('mass', 'f4'),
        ('theta', 'f4'),
        ('phi', 'f4'),
        ('vx', 'f4'),
        ('vy', 'f4'),
        ('vz', 'f4'),
        ('gen_status', 'i4'),
        ('decayed_in_tracker', 'i4'),
        ('decayed_in_calo', 'i4'),
        ('mc_index', 'i4')  # Índice original en MCParticles (para referencia)
    ])
    
    for i, part in enumerate(particles_list):
        for key in particles_array.dtype.names:
            particles_array[i][key] = part[key]
    
    # Crear array de hits
    n_hits = len(hits_list)
    hits_array = np.zeros(n_hits, dtype=[
        ('detector_type', 'i4'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('energy', 'f4'),
        ('p', 'f4'),
        ('px', 'f4'),
        ('py', 'f4'),
        ('pz', 'f4'),
        ('theta', 'f4'),
        ('phi', 'f4')
    ])
    
    # Listas para los enlaces hit-partícula
    hit_to_particle = []
    hit_weights = []
    
    for i, hit in enumerate(hits_list):
        for key in hits_array.dtype.names:
            hits_array[i][key] = hit[key]
        
        # Guardar enlaces y pesos (rellenando con -1 si hay menos de 5)
        indices = hit['gen_particle_indices'][:5]  # Máximo 5
        weights = hit['gen_weights'][:5]
        
        # Rellenar con -1 si es necesario
        indices += [-1] * (5 - len(indices))
        weights += [-1.0] * (5 - len(weights))
        
        hit_to_particle.append(indices)
        hit_weights.append(weights)
    
    return particles_array, hits_array, np.array(hit_to_particle), np.array(hit_weights)


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_nn_format.py input_file output_file [CLIC] [max_events]")
        print("Example: python convert_to_nn_format.py input.root output.npz True 1000")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    is_clic = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    max_events = int(sys.argv[4]) if len(sys.argv) > 4 else -1  # -1 = procesar todos
    
    # Configurar logging
    log_filename = setup_logging(output_file)
    
    logging.info("="*70)
    logging.info("CONVERSION TO NEURAL NETWORK FORMAT STARTED")
    logging.info("="*70)
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"CLIC detector: {is_clic}")
    logging.info(f"Max events: {max_events if max_events > 0 else 'ALL'}")
    logging.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*70)
    flush_all()
    
    reader = root_io.Reader(input_file)
    
    # Listas para acumular eventos
    all_particles = []
    all_hits = []
    all_hit_to_particle = []
    all_hit_weights = []
    event_boundaries = [0]  # Índices donde empieza cada evento
    
    # Contar eventos totales si es posible
    events_collection = reader.get("events")
    try:
        total_events = events_collection.size()
        if max_events > 0 and max_events < total_events:
            total_events = max_events
        logging.info(f"Total events to process: {total_events}")
    except:
        total_events = max_events if max_events > 0 else None
        logging.info(f"Total events: Unknown (will process {max_events if max_events > 0 else 'all'})")
    
    debug = False
    n_events_processed = 0
    start_time = time.time()
    last_log_time = start_time
    
    logging.info("\nStarting event processing...")
    logging.info("-" * 70)
    flush_all()
    
    for i, event in enumerate(events_collection):
        # Verificar si hemos alcanzado el límite de eventos
        if max_events > 0 and i >= max_events:
            logging.info(f"\nReached maximum number of events ({max_events}), stopping...")
            break
        
        current_time = time.time()
        # Log cada 100 eventos o cada 60 segundos
        if (i + 1) % 100 == 0 or (current_time - last_log_time) >= 60:
            if total_events:
                progress_pct = (i + 1) / total_events * 100
                time_est = estimate_time_remaining(start_time, i + 1, total_events)
                logging.info(f"Progress: {i + 1}/{total_events} eventos ({progress_pct:.1f}%) - {time_est}")
            else:
                elapsed = str(timedelta(seconds=int(current_time - start_time)))
                rate = (i + 1) / (current_time - start_time)
                logging.info(f"Processed {i + 1} events - Elapsed: {elapsed}, Rate: {rate:.2f} ev/s")
            flush_all()  # Forzar escritura inmediata
            last_log_time = current_time
        
        # Activar debug solo para los primeros eventos
        debug = (i < 2)
        
        if debug:
            logging.info(f"\n{'='*60}")
            logging.info(f"EVENT {i}")
            logging.info(f"{'='*60}")
        
        # Procesar partículas generadas
        particles_list, genpart_map = process_gen_particles(event, debug)
        
        if len(particles_list) == 0:
            logging.warning(f"Event {i}: No generated particles found, skipping event")
            flush_all()
            continue
        
        # Procesar hits de todos los detectores
        hits_list = []
        
        # Tracks (Inner Tracker)
        if debug:
            logging.info("  Processing Inner Tracker...")
        track_hits = process_tracks(event, genpart_map, particles_list, is_clic, debug)
        hits_list.extend(track_hits)
        if debug:
            logging.info(f"    -> {len(track_hits)} track hits")
        
        # Calorimeter hits (ECAL, HCAL)
        if debug:
            logging.info("  Processing Calorimeters...")
        gen_part_coll = event.get("MCParticles")
        calo_hits = process_calo_hits(event, genpart_map, gen_part_coll, is_clic, debug, log_progress=debug)
        hits_list.extend(calo_hits)
        if debug:
            logging.info(f"    -> {len(calo_hits)} calorimeter hits")
        
        # Muon hits
        if debug:
            logging.info("  Processing Muon system...")
        muon_hits = process_muon_hits(event, genpart_map, debug)
        hits_list.extend(muon_hits)
        if debug:
            logging.info(f"    -> {len(muon_hits)} muon hits")
        
        if len(hits_list) == 0:
            logging.warning(f"Event {i}: No hits found in event, skipping")
            flush_all()
            continue
        
        # Convertir a arrays
        particles_array, hits_array, hit_to_particle, hit_weights = convert_to_arrays(
            particles_list, hits_list
        )
        
        # Guardar para este evento
        all_particles.append(particles_array)
        all_hits.append(hits_array)
        all_hit_to_particle.append(hit_to_particle)
        all_hit_weights.append(hit_weights)
        event_boundaries.append(event_boundaries[-1] + len(hits_array))
        
        n_events_processed += 1
        
        if debug:
            logging.info(f"\nEvent {i} summary:")
            logging.info(f"  - Particles: {len(particles_array)}")
            logging.info(f"  - Hits: {len(hits_array)}")
            logging.info(f"    * Tracks: {np.sum(hits_array['detector_type'] == 0)}")
            logging.info(f"    * ECAL: {np.sum(hits_array['detector_type'] == 1)}")
            logging.info(f"    * HCAL: {np.sum(hits_array['detector_type'] == 2)}")
            logging.info(f"    * Muon: {np.sum(hits_array['detector_type'] == 3)}")
    
    # Concatenar todos los eventos
    logging.info("\n" + "="*70)
    logging.info(f"Combining {n_events_processed} events...")
    
    all_particles_concat = np.concatenate(all_particles)
    all_hits_concat = np.concatenate(all_hits)
    all_hit_to_particle_concat = np.concatenate(all_hit_to_particle)
    all_hit_weights_concat = np.concatenate(all_hit_weights)
    
    logging.info(f"  - Total particles: {len(all_particles_concat)}")
    logging.info(f"  - Total hits: {len(all_hits_concat)}")
    
    # Guardar en formato NPZ
    logging.info(f"\nSaving to {output_file}...")
    save_start = time.time()
    
    np.savez_compressed(
        output_file,
        particles=all_particles_concat,
        hits=all_hits_concat,
        hit_to_particle=all_hit_to_particle_concat,
        hit_weights=all_hit_weights_concat,
        event_boundaries=np.array(event_boundaries),
        n_events=n_events_processed
    )
    
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    
    logging.info(f"File saved in {save_time:.1f} seconds")
    
    logging.info("\n" + "="*70)
    logging.info("CONVERSION COMPLETE")
    logging.info("="*70)
    logging.info(f"Total events processed: {n_events_processed}")
    logging.info(f"Total particles: {len(all_particles_concat)}")
    logging.info(f"Total hits: {len(all_hits_concat)}")
    logging.info(f"  - Tracks: {np.sum(all_hits_concat['detector_type'] == 0)}")
    logging.info(f"  - ECAL: {np.sum(all_hits_concat['detector_type'] == 1)}")
    logging.info(f"  - HCAL: {np.sum(all_hits_concat['detector_type'] == 2)}")
    logging.info(f"  - Muon: {np.sum(all_hits_concat['detector_type'] == 3)}")
    logging.info(f"Output file: {output_file}")
    import os
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    logging.info(f"File size: {file_size_mb:.2f} MB")
    logging.info(f"Total processing time: {str(timedelta(seconds=int(total_time)))}")
    logging.info(f"Average rate: {n_events_processed/total_time:.2f} events/second")
    logging.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("\nData structure:")
    logging.info("  - particles: array with particle properties (PID, charge, momentum, etc.)")
    logging.info("  - hits: array with hit properties (position, energy, detector type, etc.)")
    logging.info("  - hit_to_particle: mapping from hits to particles (up to 5 per hit)")
    logging.info("  - hit_weights: weights for each hit-particle association")
    logging.info("  - event_boundaries: indices marking event boundaries in hit array")
    logging.info("\nDetector types:")
    logging.info("  0 = Inner Tracker")
    logging.info("  1 = ECAL")
    logging.info("  2 = HCAL")
    logging.info("  3 = Muon Tracker")
    logging.info("="*70)
    flush_all()


if __name__ == "__main__":
    main()
