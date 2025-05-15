import tkinter as tk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from minisom import MiniSom
import random
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

data = load_digits()
X = data.data      # Shape: (1797, 64)
y = data.target
#
# from sklearn.datasets import load_wine
# data = load_wine()
# X = data.data      # Shape: (178, 13)
# y = data.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the main window
root = tk.Tk()
root.title("Dimensionality Reduction Techniques")

# Frames
control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# Method Selection
method_label = tk.Label(control_frame, text="Select Technique:")
method_label.pack()

method_var = tk.StringVar()
method_combobox = ttk.Combobox(control_frame, textvariable=method_var)
method_combobox['values'] = ['PCA', 't-SNE', 'UMAP', 'Isomap', 'SOM', 'Autoencoder', 'Autoencoder + t-SNE']
method_combobox.pack()
method_combobox.current(0)

# Frame for parameters
param_frame = tk.Frame(control_frame)
param_frame.pack(pady=10)

# Dictionary to store parameter entries
param_entries = {}


def create_param(name, default=''):
    """Helper function to create input fields dynamically."""
    label = tk.Label(param_frame, text=name)
    label.pack()
    entry = tk.Entry(param_frame)
    entry.insert(0, default)
    entry.pack()
    param_entries[name] = entry


def update_parameters(event):
    """Update available parameters based on selected method."""
    for widget in param_frame.winfo_children():
        widget.destroy()

    method = method_var.get()
    param_entries.clear()

    if method == 'PCA':
        create_param('n_components', default='2')

    elif method == 't-SNE':
        create_param('n_components', default='2')
        create_param('perplexity', default='30')

    elif method == 'UMAP':
        create_param('n_components', default='2')
        create_param('n_neighbors', default='15')
        create_param('min_dist', default='0.1')

    elif method == 'Isomap':
        create_param('n_components', default='2')
        create_param('n_neighbors', default='5')

    elif method == 'SOM':
        create_param('grid_size', default='10')
        create_param('sigma', default='1.0')
        create_param('learning_rate', default='0.5')

    elif method == 'Autoencoder':
        create_param('encoding_dim', default='2')
        create_param('epochs', default='50')
        create_param('batch_size', default='16')

    elif method == 'Autoencoder + t-SNE':
        create_param('encoding_dim', default='16')
        create_param('epochs', default='100')
        create_param('batch_size', default='32')
        create_param('perplexity', default='30')


method_combobox.bind('<<ComboboxSelected>>', update_parameters)

canvas = None  # To store the plot canvas


def run_method(seed=None):
    global canvas

    method = method_var.get()

    fig, ax = plt.subplots(figsize=(7, 6))

    try:
        if canvas:
            canvas.get_tk_widget().destroy()
    except tk.TclError:
        pass

    try:
        if method == 'PCA':
            n_components = int(param_entries['n_components'].get())
            pca = PCA(n_components=n_components, random_state=seed)
            X_pca = pca.fit_transform(X_scaled)
            plot_data = X_pca[:, :2]

        elif method == 't-SNE':
            n_components = int(param_entries['n_components'].get())
            perplexity = float(param_entries['perplexity'].get())
            tsne = TSNE(n_components=n_components, random_state=seed, perplexity=perplexity)
            plot_data = tsne.fit_transform(X_scaled)

        elif method == 'UMAP':
            n_components = int(param_entries['n_components'].get())
            n_neighbors = int(param_entries['n_neighbors'].get())
            min_dist = float(param_entries['min_dist'].get())
            reducer = umap.UMAP(n_components=n_components, random_state=seed,
                                n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean')
            plot_data = reducer.fit_transform(X_scaled)

        elif method == 'Isomap':
            n_components = int(param_entries['n_components'].get())
            n_neighbors = int(param_entries['n_neighbors'].get())
            isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            plot_data = isomap.fit_transform(X_scaled)

        elif method == 'SOM':
            grid_size = int(param_entries['grid_size'].get())
            sigma = float(param_entries['sigma'].get())
            learning_rate = float(param_entries['learning_rate'].get())
            som = MiniSom(x=grid_size, y=grid_size, input_len=X_scaled.shape[1],
                          sigma=sigma, learning_rate=learning_rate, random_seed=seed)
            som.train_random(X_scaled, 100)
            plot_data = np.array([som.winner(x) for x in X_scaled])

        elif method == 'Autoencoder':
            encoding_dim = int(param_entries['encoding_dim'].get())
            epochs = int(param_entries['epochs'].get())
            batch_size = int(param_entries['batch_size'].get())
            input_dim = X_scaled.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(10, activation='relu')(input_layer)
            encoded = Dense(encoding_dim, activation='linear')(encoded)
            decoded = Dense(10, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
            plot_data = encoder.predict(X_scaled)

        elif method == 'Autoencoder + t-SNE':
            encoding_dim = int(param_entries['encoding_dim'].get())
            epochs = int(param_entries['epochs'].get())
            batch_size = int(param_entries['batch_size'].get())
            perplexity = float(param_entries['perplexity'].get())

            # Autoencoder
            input_dim = X_scaled.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(64, activation='relu')(input_layer)
            encoded = Dense(encoding_dim, activation='linear')(encoded)
            decoded = Dense(64, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            autoencoder = Model(input_layer, decoded)
            encoder = Model(input_layer, encoded)
            autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

            # Apply t-SNE to Autoencoder's encoded output
            encoded_data = encoder.predict(X_scaled)
            tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
            plot_data = tsne.fit_transform(encoded_data)

        # Plotting
        scatter = ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            c=y,
            cmap='tab10',  # Use discrete colors for classification
            edgecolor='k',
            s=40
        )
        ax.set_title(f"{method} Result (Seed: {seed})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)

        # Add legend for class labels
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="best")
        ax.add_artist(legend1)

        # Embed the plot in the GUI
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    except Exception as e:
        print(f"Error: {e}")


# Run button for one time
run_button = tk.Button(control_frame, text="Run", command=run_method)
run_button.pack(pady=20)


# Button for running 30 repetitions
def run_30_reps():
    seeds = [random.randint(0, 10000) for _ in range(30)]  # List of 30 seeds
    for seed in seeds:
        run_method(seed=seed)  # Run the selected technique with a new seed for each repetition

    # Save seeds to a file for later reference
    with open("seeds_used.txt", "w") as f:
        for seed in seeds:
            f.write(f"{seed}\n")
    print(f"Seeds saved to 'seeds_used.txt'.")


run_30_button = tk.Button(control_frame, text="Run 30 Repetitions", command=run_30_reps)
run_30_button.pack(pady=10)

# Initialize with parameters of first method
update_parameters(None)

root.mainloop()
