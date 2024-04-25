file_name = "Waveforms_Ds3613_Bkg_Ch0001.root"

import uproot
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import awkward as ak

file = uproot.open('./' + file_name)

#print(file.keys())
tree = file[file.keys()[0]]
df = tree.arrays(library="pd")
#print(df.head())

print(df['Waveform'].head())


def apply_pca_to_waveforms3d(data):
    waveforms = [ak.to_numpy(wf) for wf in data]
    waveforms = np.stack(waveforms)
    scaler = StandardScaler()
    waveforms = scaler.fit_transform(waveforms.T).T
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(waveforms)
    return principal_components

def plot_pca_results3d(components):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(components[:, 0], components[:, 1], components[:, 2], alpha=0.5)
    ax.set_title('3D PCA Results for Waveforms')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.savefig('3d_pca_waveform_results.png')
    plt.show()

def apply_pca_to_waveforms(data):
    waveforms = [ak.to_numpy(wf) for wf in data]
    waveforms = np.stack(waveforms)
    scaler = StandardScaler()
    waveforms = scaler.fit_transform(waveforms.T).T
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(waveforms)
    return principal_components

def plot_pca_results(components):
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.5)
    plt.title('PCA Results for Waveforms')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig('pca_waveform_results.png')
    plt.show()

if __name__ == "__main__":
    # pca_results = apply_pca_to_waveforms(df['Waveform'])
    # plot_pca_results(pca_results)
    pca_results = apply_pca_to_waveforms3d(df['Waveform'])
    plot_pca_results3d(pca_results)