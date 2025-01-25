from Quiko_Preprocessing import samp_subband, sample_database, Init_Database
import json
import qiskit
from qiskit import QuantumCircuit
from qiskit import qpy  # for qpy serialization
import sys
import os
import argparse
import numpy as np
import random
sys.path.insert(0, os.getcwd())


def save_database_bundle(database_bundle, names, folder_list, save_directory="/data/cvcqml/common/maiol/DB"):
    """
    Save each quantum circuit in the database_bundle to .qpy files organized by folder.
    
    Args:
        database_bundle (list): List of quantum circuits.
        names (list): List of track names corresponding to each circuit.
        folder_list (list): List of folders each track belongs to (e.g., 'hit-hats' or 'shakers').
        save_directory (str): Main directory to save the organized database.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Dictionary to hold track names and file paths
    circuit_paths = {}

    # Loop through each circuit, track name, and corresponding folder
    for circuit, track_name, folder in zip(database_bundle, names, folder_list):
        # Define subfolder path based on the track's folder
        folder_path = os.path.join(save_directory, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Define file path for each circuit in its respective subfolder
        circuit_path = os.path.join(folder_path, f"{track_name}.qpy")
        
        # Save the circuit as .qpy
        with open(circuit_path, "wb") as file:
            qpy.dump(circuit, file)
        
        # Store the file path in a dictionary for reference
        circuit_paths[track_name] = circuit_path
    
    # Save circuit paths to a JSON file in the main save directory
    json_path = os.path.join(save_directory, "database_bundle_paths.json")
    with open(json_path, "w") as json_file:
        json.dump(circuit_paths, json_file)

    print(f"Database bundle saved to {save_directory}")

def load_database_bundle(load_directory="/data/cvcqml/common/maiol/DB"):
    """
    Load database_bundle from saved .qpy files organized by subfolders.
    
    Args:
        load_directory (str): Main directory containing the saved database with subfolders.
        
    Returns:
        tuple: (circuit_bundle, track_names)
            - circuit_bundle (list): List of loaded quantum circuits.
            - track_names (list): List of track names corresponding to each circuit.
    """
    # Path to JSON file containing paths for each track's .qpy file
    json_path = os.path.join(load_directory, "database_bundle_paths.json")
    
    # Load the JSON file with circuit paths
    with open(json_path, "r") as json_file:
        circuit_paths = json.load(json_file)
    
    # Lists to hold loaded circuits and track names
    circuit_bundle = []
    track_names = []
    
    # Load each circuit from its .qpy file path stored in circuit_paths
    for track_name, circuit_path in circuit_paths.items():
        # Construct the full path
        full_path = os.path.join(load_directory, circuit_path)
        
        # Load the circuit from the .qpy file
        with open(full_path, "rb") as file:
            loaded_circuit = qpy.load(file)[0]  # qpy.load returns a list of circuits
            
            # Append to lists
            circuit_bundle.append(loaded_circuit)
            track_names.append(track_name)

    print("Database bundle loaded successfully from organized subfolders.")
    return circuit_bundle, track_names


subB = {2:[1080], # this is the 2-qubit case
        3:[920, 3150], # This is for the 3-qubit case
        4:[630, 1720,  4400], # This is for the 4-qubit case
        5:[510, 1270, 2700, 6400], # This is for the 5-qubit case
        6:[400, 920, 1720, 3150, 6400], # This is for the 6-qubit case
        8:[300, 630, 1080, 1720, 2700, 4400, 7700], # This is for the 8-qubit case
        12:[200, 400, 630, 920, 1270, 1720, 2320, 3150, 4400, 6400, 9500], # This is for the 12-qubit case
        25:[100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]} # This is for the 25-qubit case


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Database Construction")
    parser.add_argument("--seed", type=int, help="Seed value for random generation")

    args = parser.parse_args()
    
    seed = args.seed 

    np.random.seed(seed)
    random.seed(seed)

    num_bands = 3
    band_mapping = subB[num_bands]

    # Directory containing drum samples
    folder = '/data/cvcqml/common/maiol' #check claps for debugging


    backend = 'aer_simulator'  # Local simulator or real device if preferred
    boundle, names, folder_list = Init_Database(num_bands, band_mapping, folder, backend)

    print('------------------------------------')
    save_database_bundle(boundle, names, folder_list)
    # database_bundle, filenames = load_database_bundle()

