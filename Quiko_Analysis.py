import csv
import numpy as np
from math import pi
import itertools
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from QuiKo_Backend import *
from Quiko_Preprocessing import *
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
import json
import os
from collections import defaultdict
import librosa
import random
import pandas as pd

#print('Loading analysis testing version...')
#print('Loading IBM Account...')

subB = {2:[1080], 
        3:[920, 3150], 
        4:[630, 1720, 4400], 
        5:[510, 1270, 2700, 6400], 
        6:[400, 920, 1720, 3150, 6400], 
        8:[300, 630, 1080, 1720, 2700, 4400, 7700], 
        12:[200, 400, 630, 920, 1270, 1720, 2320, 3150, 4400, 6400, 9500], 
        25:[100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]}

# Utility Functions and Execution Functions ---------------------------------------------
def execute_quantumcircuit_ANALY(quantumcircuit, inject_noise=False, dev='aer_simulator'):
    """
    Execute a quantum circuit with optional noise on a specified backend.
    """
    simulator = AerSimulator()
    
    if dev == 'aer_simulator':
        if inject_noise:
            provider = IBMQ.load_account()
            backend = provider.get_backend('ibmq_belem')
            noise_model = NoiseModel.from_backend(backend)
            result = qiskit.execute(quantumcircuit, simulator, noise_model=noise_model).result()
        else:
            transpiled_circuit = transpile(quantumcircuit, simulator)
            result = simulator.run(transpiled_circuit).result()
        distribution = result.get_counts()
    else:
        provider = IBMQ.get_provider(hub='ibm-q-academic', group='stanford', project='q-comp-music-gen')
        backend = provider.get_backend(dev)
        job = backend.run(transpile(quantumcircuit, backend, optimization_level=3))
        from qiskit.tools.monitor import job_monitor
        job_monitor(job)
        distribution = job.result().get_counts()

    return distribution

# Utility Functions
"""
def zero_fill(distr, qubit_count):
    keys = np.sort(list(distr.keys()))
    targ_index = np.arange(2**qubit_count)
    targ_key = [format(i, '0{}b'.format(qubit_count)) for i in targ_index]
    diff = np.setdiff1d(targ_key, keys)
    for i in diff:
        distr[i] = 0
    return distr
"""

def zero_fill(distr, qubit_count):
    """
    This is going to be a utility function for filling in the states
    that the probabilities are zero (Already wrote one just kick it...)
    """
    keys = np.sort(list(distr.keys()))
    print('keys: ',keys)
    #print(keys)
    targ_index = np.arange( 2**qubit_count )
    targ_key = [format(i, '0{}b'.format(qubit_count)) for i in targ_index]
    print('targ_key: ',targ_key)
    diff = np.setdiff1d(targ_key, keys)
    print('diff: ', diff)
    for i in diff:
        #print(i,type(i))ß
        distr[i] = 0
    #print('what is wrong: ',distr.keys(), distr)
    return distr
    #    if distr[k]

# Expressibility and Analysis Classes ---------------------------------------------
class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)

class fidelity_measure:
    def __init__(self, qubits, distr_1, distr_2):
        self.qubits = qubits
        self.distr_1 = distr_1
        self.distr_2 = distr_2

    def measured_qstate(self):

        
        # Zero-fill the distributions to include all states
        distr_1 = zero_fill(self.distr_1, self.qubits)
        distr_2 = zero_fill(self.distr_2, self.qubits)
        
        print(distr_1)
        print(distr_2)

        coefficients_db  = {key: np.sqrt(value / n_shots) for key, value in distr_2.items()}
        coefficients_test  = {key: np.sqrt(value / n_shots) for key, value in distr_1.items()}

        print(coefficients_db)

        fidelity = sum(coefficients_test.get(key, 0) * coefficients_db.get(key, 0) for key in set(coefficients_test.keys()).union(coefficients_db.keys()))


        return fidelity

    def fidelity_meas(self):
        fidelity = self.measured_qstate()
        return fidelity

class Expressibility:
    def __init__(self, num_bands, timbre_qubits, spine_qubits, qubits, database_bundle):
        self.num_bands = num_bands
        self.spine_qubits = spine_qubits
        self.timbre_qubits = timbre_qubits
        self.qubits = qubits
        self.dbb = database_bundle

    def Haar_meas(self, fidelities):
        uniq_fid = len(fidelities)
        prob = 1 / uniq_fid
        return {f: prob for f in fidelities}

    def random_meas(self, M_samp=200, encoding_method='pkbse', big_circ=False, dev='aer_simulator'):
        random_meas = {}
        for M in range(M_samp):
            random_matrix = []
            for band in range(self.num_bands):
                angles = [round(angle, 3) for angle in sin_sampler.rvs(size=1)] + [round(2 * np.pi * np.random.uniform(), 3) for _ in range(2)]
                random_matrix.append(angles)

            # Encoding and Circuit Construction
            simulator = AerSimulator()
            qc = QuantumCircuit(self.qubits)
            if big_circ:
                bundle_circuit = transpile(qc, simulator)
                result = simulator.run(bundle_circuit).result()
                random_meas = result.get_counts()
            else:
                encoded_circuit = QuantumCircuit(self.qubits)
                transpiled_circuit = transpile(encoded_circuit, simulator)
                result = simulator.run(transpiled_circuit).result()
                random_meas = result.get_counts()
                
        return random_meas, self.dbb

    def Express(self, distr_1, dev, fid_meas=False):
        window_Rmeas = []; window_Hmeas = []
        random_meas, haar_meas = self.random_meas(M_samp=50, encoding_method='pkbse', dev=dev)
        
        if fid_meas:
            return np.sort(list(haar_meas.keys())), haar_meas
        
        window_Hmeas.append(self.Haar_meas(random_meas))
        window_Rmeas.append(random_meas)

        return window_Rmeas, window_Hmeas

    def KLD(self, random_meas, haar_meas):
        states = random_meas.keys()
        return np.sum([random_meas[i] * np.log(random_meas[i] / haar_meas[i]) for i in states])

def main_analysis():
    num_bands = 3
    spine_qubits = 3
    timbre_qubits = 3
    qubits = timbre_qubits + spine_qubits
    p = Expressibility(num_bands, timbre_qubits, spine_qubits, qubits, {})
    distr_1, database_bundle = p.random_meas(M_samp=50, encoding_method='pkbse', big_circ=True, dev='aer_simulator')
    rand, haar = p.Express(distr_1, 'aer_simulator')
    return [rand, haar]

def ph_elements(test_sample_path, num_bands, band_mapping):
    """
    Plot percussive and harmonic elements
    """
    print(f"Extracting features from test sample: {test_sample_path}")
    test_audio, fs = librosa.load(test_sample_path, sr=44100)
    subbands = subband_gen(test_audio, fs, num_bands, band_mapping)
    print(subbands)
    
    # Separate each subband into harmonic and percussive components
    meas_h, meas_p = librosa.effects.hpss(subbands[0])
    
    # Create the plots directory if it does not exist
    plots_dir = './plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot harmonic and percussive measurements
    plt.figure(figsize=(10, 6))
    plt.plot(meas_h, color='blue', label='Harmonic Measurement')
    plt.plot(meas_p, color='orange', label='Percussive Measurement')
    plt.title(f'Low Subband: Harmonic vs Percussive')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(plots_dir, f'subband_0_harmonic_vs_percussive.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()  # Close the plot to avoid overlap with the next one
    
def sample_wav_files_from_json(file_path, size_random = 8,seed = None):
    
    if seed is not None:
        np.random.seed(seed)
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    fidelity_grid = data["fidelity_grid"]
    prob = data["output_probs_test"]

    # Use numpy to sample subdivisions based on probabilities
    sampled_subdivisions = np.random.choice(
        list(fidelity_grid.keys()), size=size_random, replace=True, p=prob
    )

    sampled_files = {}
    print("Probabilities: ", prob)
    print("Sampled subdivisions: ", sampled_subdivisions)
    # Track available files for each subdivision
    available_files = {key: [item[0] for item in subdivision] for key, subdivision in fidelity_grid.items()}
    
    for subdivision in sampled_subdivisions:
        if available_files[subdivision]:
            # Assign the first file from the available list and remove it
            sampled_file = available_files[subdivision].pop(0)
            if subdivision not in sampled_files:
                sampled_files[subdivision] = []
            sampled_files[subdivision].append(sampled_file)

    sampled_files = {key: sampled_files[key] for key in sorted(sampled_files.keys(), key=int)}
    return sampled_files, prob, fidelity_grid

def plot_top_similarities(*dictionaries, plots_dir):
    """
    Combine multiple dictionaries, calculate summed similarities for each category,
    and plot the top 3 values per category with unique colors per category.

    Args:
        *dictionaries: Variable number of dictionaries to combine.
    """
        
    # Combine dictionaries
    combined_sums = defaultdict(float)
    for dictionary in dictionaries:
        for key, value_list in dictionary.items():
            for name, similarity in value_list:
                combined_sums[(key, name)] += similarity

    # Find top 3 for each category
    top_3_per_category = defaultdict(list)
    for (category, name), total_similarity in combined_sums.items():
        top_3_per_category[category].append((name, total_similarity))

    # Sort by similarity and take top 3
    for category in top_3_per_category:
        top_3_per_category[category].sort(key=lambda x: x[1], reverse=True)
        top_3_per_category[category] = top_3_per_category[category][:3]

    # Prepare data for plotting
    categories = sorted(top_3_per_category.keys())
    x_labels = []
    y_values = []
    colors = []
    category_colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))  # Generate unique colors

    for category_idx, category in enumerate(categories):
        for name, similarity in top_3_per_category[category]:
            x_labels.append(f"{category}-{name}")
            y_values.append(similarity)
            colors.append(category_colors[category_idx])  # Assign color to category

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(x_labels, y_values, color=colors)
    plt.xlabel("File Name")
    plt.ylabel("Total Similarity")
    plt.title("Top 3 Similarities per Subdivision Across all songs")
    plt.xticks(rotation=90, fontsize=12)
    plt.tight_layout()

    # Save the plot
    plot_filename = os.path.join(plots_dir, f'Total_Sample_Fidelity.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()  # Close the plot to avoid overlap with the next one

def plot_probabilities(probabilities, file=None, plots_dir = None):
    """
    Plot summed probabilities from a list of lists.

    Args:
        probabilities (list of lists or array): A list or array where each element contains probabilities.
        file (str, optional): File name for saving the plot.
    """
    if isinstance(probabilities[0], list):
        # Calculate the average probabilities if it's a list of lists
        summed_probabilities = np.sum(probabilities, axis=0)
        summed_probabilities = summed_probabilities / len(probabilities)
        Flag = True
    else:
        # Handle single probability list or other errors
        summed_probabilities = probabilities
        Flag = False

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(summed_probabilities)), summed_probabilities)

    if Flag:
        plt.xlabel("Index")
        plt.ylabel("Summed Probability")
        plt.title("Summed Probabilities Across Different Songs")
        # Save the plot
        plot_filename = os.path.join(plots_dir, 'Probabilities.png')
    else:
        plt.xlabel("Index")
        plt.ylabel("Probability")
        plt.title(f"Probabilities Across Subdivisions for {file}")
        # Save the plot with the file name (if provided)
        if file:
            file = file[:-4] if file.endswith('.png') else file
            plot_filename = os.path.join(plots_dir, f'Probabilities_{file}.png')
        else:
            plot_filename = os.path.join(plots_dir, 'Probabilities_Unknown.png')

    # Save and close the plot
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()

def plot_samples_plot(data, file=None, plots_dir = None):
    # Prepare data for the plot
    # Directory for saving plots

        
    keys = list(data.keys())
    labels = [item[0] for key in keys for item in data[key][:3]]
    similarities = [item[1] for key in keys for item in data[key][:3]]
    categories = np.repeat(keys, 3)

    # Plotting
    plt.figure(figsize=(12, 8))
    for category in keys:
        category_values = [item[1] for item in data[category][:3]]
        plt.bar(
            [f"{category}-{item[0]}" for item in data[category][:3]],
            category_values,
            label=f"Subdivision {category}",
        )

    plt.xlabel("File Name")
    plt.ylabel("Similarity")
    plt.title("Top 3 Similarities for Each Subdivision")
    plt.xticks(rotation=90, fontsize=12)
    plt.legend()
    plt.tight_layout()
    file = file[:-4] if file.endswith('.png') else file
    plot_filename = os.path.join(plots_dir, f'Samples_{file}.png')

    # Save and close the plot
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()
    

def count_wav_files(data, plots_dir):
    counts_per_subdivision = defaultdict(lambda: defaultdict(int))
    # Iterate through the list of dictionaries
    for dictionary in data:
        for subdivision, wav_list in dictionary.items():
            for wav_file, _ in wav_list:
                counts_per_subdivision[subdivision][wav_file] += 1

    # Keep only the top 3 .wav files for each subdivision
    top_counts_per_subdivision = {}
    for subdivision, wav_counts in counts_per_subdivision.items():
        sorted_counts = sorted(wav_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_counts_per_subdivision[subdivision] = dict(sorted_counts)

    categories = sorted(top_counts_per_subdivision.keys())

    x_labels = []
    y_values = []
    colors = []
    category_colors = plt.cm.tab10(np.linspace(0, 1, len(top_counts_per_subdivision.keys())))  # Generate unique colors
    for category_idx, category in enumerate(categories):
      for name, similarity in top_counts_per_subdivision[category].items():
          x_labels.append(f"{category}-{name}")
          y_values.append(similarity)
          colors.append(category_colors[category_idx])  # Assign color to category

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(x_labels, y_values, color=colors)
    plt.xlabel("File Name")
    plt.ylabel("Total Similarity")
    plt.title("Top 3 most common Samples per Subdivision Across all songs")
    plt.xticks(rotation=90, fontsize=12)
    plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in category_colors],
                labels=[f"Subdivision {n}" for n in range(len(top_counts_per_subdivision.keys()))], title="Subdivisions", bbox_to_anchor=(1.05, 1), loc='upper left')


    plot_filename = os.path.join(plots_dir, f'Num_samples.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot saved as {plot_filename}")
    plt.close() 



    
if __name__ == "__main__":
    TEST_SAMPLE_PATH = './grids/dataset/classical'
    probabilities = []
    total = []
    input_folder = './grids/dataset'
    output_folder = './plots/dataset'   
    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        # Extract the genre from the relative path
        genre = os.path.basename(relative_path)
        if not genre:  # If the folder itself is the input root
            continue

        for letter in range(len(relative_path)):
            if relative_path[letter] == "/":
                layers = relative_path[:letter]

        for file in files:
            file_path = os.path.join(root, file)
            print("File: ",file)

            # Create the output folder hierarchy: Layers_X/Genre
            layer_folder = os.path.join(output_folder, f"{layers}", genre)
            os.makedirs(layer_folder, exist_ok=True)

            # Process the file
            a, probab, grid = sample_wav_files_from_json(file_path, 8, 42)
            print("Grid ", grid)

            probabilities.append(probab)
            plot_probabilities(probab, file, layer_folder)
            plot_samples_plot(grid, file, layer_folder)
            total.append(grid)

            print("Sampled Grid: ",a)
            print('-------------------------------------------------------------------------------')
        if not dirs:
            print("Total: ", total)

            count_wav_files(total,layer_folder)
            plot_probabilities(probabilities, plots_dir=layer_folder)
            plot_top_similarities(*total, plots_dir=layer_folder)
        