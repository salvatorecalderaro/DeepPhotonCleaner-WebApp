import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from astropy.io import fits

tmp_dir = "tmp"
os.makedirs(tmp_dir, exist_ok=True)
dpi = 1000

def plot_curve(obs_id, filename, nbins, grid, binned_data):
    bin_centers = (grid[:-1] + grid[1:]) / 2
    path = os.path.join(tmp_dir, f"{filename}_{nbins}_noisy.png")
    
    plt.figure(figsize=(15, 6))
    plt.title(f"Observation {obs_id} - {filename}, Noisy Curve N.Bins={nbins}")
    plt.plot(bin_centers, binned_data[0], 'o-', label='Original curve', color='black')
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    
    return path


def plot_results(obs_id, filename, nbins, grid, data, cleaned_curve, good_bins):
    bin_centers = (grid[:-1] + grid[1:]) / 2

    plt.figure(figsize=(15, 6))
    plt.plot(bin_centers, data[0], 'o-', label='Original curve', color='blue')
    plt.plot(bin_centers, cleaned_curve, 'o--', label='Clean curve (DPC)', color='green')

    # Highlight good bins region
    if good_bins is not None and len(good_bins) > 0:
        t_start = float(bin_centers[min(good_bins)])
        t_end = float(bin_centers[max(good_bins)])
        plt.axvspan(t_start, t_end, color='orange', alpha=0.3, label='Refererence Part')

    # Add titles and labels
    plt.title(f"DeepPhotonCleaner - Observation {obs_id}  {filename}, N.Bins={nbins}")
    plt.xlabel("Time")
    plt.ylabel("Counts")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc = 'best')
    
    path = os.path.join(tmp_dir, f"{filename}_{nbins}_DPC.png")
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path

def convert_endian(df):
    for col in df.columns:
        dtype = df[col].dtype
        if dtype.byteorder == ">" and np.issubdtype(dtype, np.number):
            swapped = df[col].values.byteswap()
            df[col] = swapped.view(swapped.dtype.newbyteorder()).copy()
    return df


def plot_spectra(obs_id, filename, nbins,ori_path,sas_path,clean_path):
    bin_width = 0.1
    bins = np.arange(0.3, 15.0 + bin_width, bin_width)
    with fits.open(ori_path) as hdul:
        data = hdul[1].data
    data = pd.DataFrame(data)
    energies = np.array(data["PI"]) / 1000
    
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, edges = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm = counts / (n_photons * bin_width)

    
    with fits.open(sas_path) as hdul:
        data = hdul[1].data
    data = pd.DataFrame(data)
    data = convert_endian(data)
    energies = np.array(data["PI"]) / 1000
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, edges = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm_allevc = counts / (n_photons * bin_width)

    # DPC cleaned
    with fits.open(clean_path) as hdul:
        data = hdul[1].data
    df = pd.DataFrame(data)
    df = convert_endian(df)
    df = df[(df["IS_FLAIR"] == 0)]
    energies = np.array(df["PI"]) / 1000
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, edges = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm_dpc = counts / (n_photons * bin_width)

    plt.figure(figsize=(10, 6))
    bin_centers = 0.5*(edges[:-1]+edges[1:])
    plt.title(f"Spectra Comparison for Observation {obs_id} - {filename}")
    plt.plot(bin_centers, counts_norm, label='Raw', color='black', alpha=0.7)
    plt.plot(bin_centers, counts_norm_allevc, label='SAS', color='blue', alpha=0.7)
    plt.plot(bin_centers, counts_norm_dpc, label=f"fDPC (N.Bins={nbins})", color='green', alpha=0.7)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Normalized Counts (per keV)")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
    path = f"tmp/{filename}_{nbins}_spectra_{nbins}.png"
    plt.savefig(path, dpi=dpi)
    plt.close()
    return path