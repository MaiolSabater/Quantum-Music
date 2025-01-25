import os
import json
import qiskit
from qiskit import QuantumCircuit
from qiskit import qpy  # for qpy serialization
import sys
import librosa
import qiskit.utils 
from qiskit_aer import AerSimulator, Aer
from qiskit import transpile

from QuiKo_Backend import Encoding_matrix, QuikoCircuit
from Quiko_Feature_extraction import preproc
from Quiko_Preprocessing import Init_Database, execute_quantumcircuit, subband_gen
from Quiko_Analysis import fidelity_measure
from DB_construction import load_database_bundle

import argparse
import numpy as np
import random
from collections import defaultdict

sys.path.insert(0, os.getcwd())


# Parameters for audio processing and quantum setup
NUM_BANDS = 3  # Number of frequency bands
SPINE_QUBITS = 3  # Number of subdivisions for quantum encoding
ENCODING_METHOD = 'pkbse'  # Encoding method for circuits: 'pkbse' or 'static'


subB = {2:[1080], # this is the 2-qubit case
        3:[920, 3150], # This is for the 3-qubit case
        4:[630, 1720,  4400], # This is for the 4-qubit case
        5:[510, 1270, 2700, 6400], # This is for the 5-qubit case
        6:[400, 920, 1720, 3150, 6400], # This is for the 6-qubit case
        8:[300, 630, 1080, 1720, 2700, 4400, 7700], # This is for the 8-qubit case
        12:[200, 400, 630, 920, 1270, 1720, 2320, 3150, 4400, 6400, 9500], # This is for the 12-qubit case
        25:[100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]} # This is for the 25-qubit case

BAND_MAPPING = subB[NUM_BANDS]

def preprocess_test_sample(test_sample_path, num_bands, band_mapping, spine_qubits):
    """
    Preprocess the test sample by extracting features.
    """
    print(f"Extracting features from test sample: {test_sample_path}")
    test_audio, fs = librosa.load(test_sample_path, sr=44100)
    subbands = subband_gen(test_audio, fs, num_bands, band_mapping)
    print(subbands)
    harmonic_subbands = []
    percussive_subbands = []
    for subband in subbands:
        # Separate each subband into harmonic and percussive components
        meas_h, meas_p = librosa.effects.hpss(subband)
        
        harmonic_subbands.append(meas_h)
        percussive_subbands.append(meas_p)
    # Assuming `feature_map_list` will store the feature maps for each subband
    feature_map_list = []

    # For each subband, process the harmonic and percussive data
    for i, (harm, perc) in enumerate(zip(harmonic_subbands, percussive_subbands)):
        
        # Generate feature map for the current subband
        feature_map = preproc(perc, harm, spine_qubits, fs)
        
        # Convert feature_map dictionary to a list of lists (8x3) for the current subband
        subband_features = []
        for qubit_key in sorted(feature_map.keys()):  # Sort to ensure consistent ordering
            harm_est = feature_map[qubit_key]['harm_est']
            spec_cent = feature_map[qubit_key]['spec_cent']
            perc_est = feature_map[qubit_key]['perc_est']
            subband_features.append([harm_est, spec_cent, perc_est])
        
        # Add the current subband's 8x3 feature list to the main list
        feature_map_list.append(subband_features)

    # feature_map_list should now be a 3x8x3 list (3 subbands, 8 qubits, 3 components)
    print("Feature extraction complete.")
    return feature_map_list


def encode_test_sample(feature_map, ENCODING_METHOD, NUM_BANDS, SPINE_QUBITS, layers):
    """
    Encodes the extracted features of the test sample into a quantum circuit.
    """
    print("Encoding test sample into a quantum circuit...")
    # Encode test sample features into quantum matrix
    encoding_matrix = Encoding_matrix(feature_map, 2**NUM_BANDS, NUM_BANDS, NUM_BANDS)
    if ENCODING_METHOD == 'pkbse':
        encoding_matrix = encoding_matrix.pkbse_encoding_matrix()
    else:
        encoding_matrix = encoding_matrix.static_encoding_matrix()
    
    # Build quantum circuit
    print('Encoding Matrix')
    #print(encoding_matrix)
    qc = QuikoCircuit(NUM_BANDS, SPINE_QUBITS, NUM_BANDS + SPINE_QUBITS, encoding_matrix, Encoding_method=1)
    test_qc = qc.Quantum_Circuit(Encoding_method=1, n_layers = layers)

    print("Quantum circuit for test sample created.")
    print(test_qc)
    return test_qc

def compare_test_with_database(test_qc, database_bundle, tracks_names, NUM_BANDS, file, layers):
    """
    Compare the test sample quantum circuit with the database using fidelity measures.
    Produces a grid where rows are subdivisions and columns are the top 10 most similar database tracks.
    """
    print("Executing quantum circuits for test sample and database...")

    backend = Aer.get_backend('aer_simulator')

    # Save the statevector for the test circuit with a unique label
    test_qc.save_statevector(label="test_statevector")

    transpiled_circuit_test = transpile(test_qc, backend)

    job = backend.run(transpiled_circuit_test, seed_simulator=42)
    result = job.result()

    # Retrieve the statevector using the unique label
    coefficients_test = result.data()["test_statevector"]
    coefficients_test = np.asarray(coefficients_test)
    coefficients_test = coefficients_test.reshape(-1, 2**NUM_BANDS).tolist()

    output_probs_test = np.abs(coefficients_test) ** 2
    output_probs_test = sum(output_probs_test)
    print('Test coeff computed')

    db_coefficients = {}
    for track_name, db_qc in zip(tracks_names, database_bundle):
        # Save the statevector for each database circuit with a unique label
        db_qc.save_statevector(label=f"db_statevector_{track_name + file + str(layers)}")
        transpiled_circuit_db = transpile(db_qc, backend)

        job_db = backend.run(transpiled_circuit_db, seed_simulator=42)
        result_db = job_db.result()

        # Retrieve the statevector using the unique label
        coefficients_db = result_db.data()[f"db_statevector_{track_name + file + str(layers)}"]
        coefficients_db = np.asarray(coefficients_db)
        output_probs_db = np.abs(coefficients_db) ** 2

        db_coefficients[track_name] = coefficients_db

    fidelity_grid = {}
    probs_subd = []
    for subdivision in range(2**NUM_BANDS):
        fidelity_grid[subdivision] = []

        for name_track, db_coefficient in db_coefficients.items():
            fidelity = sum(db_coefficient * coefficients_test[subdivision])
            fidelity_p = np.abs(fidelity) ** 2
            fidelity_grid[subdivision].append((name_track, fidelity_p))

        fidelity_grid[subdivision] = sorted(fidelity_grid[subdivision], key=lambda x: x[1], reverse=True)[:10]

    aggregated_values = {key: sum(value[1] for value in values) for key, values in fidelity_grid.items()}
    print(aggregated_values)

    return fidelity_grid, output_probs_test



def exec_multiple(NUM_BANDS, BAND_MAPPING, SPINE_QUBITS, ENCODING_METHOD, database_bundle, track_names, input_folder, output_folder, layers):

    for root, dirs, files in os.walk(input_folder):
        
        # Calculate the relative path from the input folder
        relative_path = os.path.relpath(root, input_folder)

        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            print(file_path)

            # Preprocess and encode the test sample
            feature_map = preprocess_test_sample(file_path, NUM_BANDS, BAND_MAPPING, SPINE_QUBITS)
            test_qc = encode_test_sample(feature_map, ENCODING_METHOD, NUM_BANDS, SPINE_QUBITS, layers)
            
            # Generate fidelity grid and probabilities
            fidelity_grid, output_probs_test = compare_test_with_database(test_qc, database_bundle, track_names, NUM_BANDS, file, layers)

            data_to_save = {
                "fidelity_grid": fidelity_grid,
                "output_probs_test": output_probs_test.tolist()
            }

            # Create the folder structure: Layers/Genre
            genre = os.path.basename(root)  # Assuming genre is the immediate parent folder
            layer_folder = os.path.join(output_folder, f"Layers_{layers}", genre)
            os.makedirs(layer_folder, exist_ok=True)

            # Save the output file in the appropriate folder
            output_file_path = os.path.join(layer_folder, f"{file}")
            print(output_file_path)

            with open(output_file_path[:-4], "w") as f:
                json.dump(data_to_save, f, indent=4)

            print('All right')
            print("Saved using:", file)

def exec_single(TEST_SAMPLE_PATH, NUM_BANDS, BAND_MAPPING, SPINE_QUBITS, ENCODING_METHOD, database_bundle, track_names):
    # Preprocess and encode the test sample
    feature_map = preprocess_test_sample(TEST_SAMPLE_PATH, NUM_BANDS, BAND_MAPPING, SPINE_QUBITS)
    test_qc = encode_test_sample(feature_map, ENCODING_METHOD, NUM_BANDS, SPINE_QUBITS)
    
    fidelity_grid, output_probs_test = compare_test_with_database(test_qc, database_bundle, track_names, NUM_BANDS)

    data_to_save = {
    "fidelity_grid": fidelity_grid,
    "output_probs_test": output_probs_test.tolist()
    }
    # Save the fidelity grid to a JSON file
    with open("./grids/fidelity_grid_" + TEST_SAMPLE_PATH[9:-4], "w") as f:
        json.dump( data_to_save, f, indent=4)

    print('All right')
    print("Saved using:", TEST_SAMPLE_PATH)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Database Construction")
    parser.add_argument("--seed", type=int, help="Seed value for random generation")

    args = parser.parse_args()
    
    seed = args.seed 

    np.random.seed(seed)
    random.seed(seed)
    
    #SAMPLE_PATH = './tracks/pianos_cut.wav'
    database_bundle, track_names = load_database_bundle()
    
    input_folder = './tracks/dataset'
    output_folder = './grids/dataset'
    layers = 3
    for layer in range(1,layers + 1):
        exec_multiple(NUM_BANDS, BAND_MAPPING, SPINE_QUBITS, ENCODING_METHOD, database_bundle, track_names, input_folder, output_folder, layer)

    #exec_single(SAMPLE_PATH, NUM_BANDS, BAND_MAPPING, SPINE_QUBITS, ENCODING_METHOD, database_bundle, track_names)