from astropy.io import fits
import pandas as pd
import numpy as np
import random
import os
import torch
import platform
import cpuinfo
from scipy.ndimage import median_filter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import streamlit as st
from astropy.table import Table

seed = 2025
epochs = 50
alpha = 1.0
beta = 0.05 
lr = 1e-3
mini_batch_size = 16
dpi = 1000
weight_time = 0.3
weight_energy = 0.7
window_size = 16
stride = 8
lambda_reg = 1e-5   

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def identify_device():
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_name = (
            torch.cuda.get_device_name()
            if device.type == "cuda"
            else cpuinfo.get_cpu_info()["brand_raw"]
        )
    if device.type == "cuda":
        set_seed(seed)
    return device, dev_name


def convert_endian(df):
    for col in df.columns:
        dtype = df[col].dtype
        if dtype.byteorder == ">" and np.issubdtype(dtype, np.number):
            swapped = df[col].values.byteswap()
            df[col] = swapped.view(swapped.dtype.newbyteorder()).copy()
    return df

def read_fits(path):
    with fits.open(path) as hdul:
        data = hdul[1].data         
        header = hdul[1].header
        #exptime = header.get('ONTIME')  # get exposure time
    glowcurvedata = pd.DataFrame(data)
    glowcurvedata = convert_endian(glowcurvedata)
    return glowcurvedata

def bin_data(glowcurvedata, nbins):
    times = glowcurvedata["TIME"]
    min_t = times.min()
    max_t = times.max()
    grid = np.linspace(min_t, max_t, nbins + 1)
    binned = pd.cut(times, bins=grid, include_lowest=True, right=True)
    counts = binned.value_counts(sort=False)
    energy_mean = glowcurvedata.groupby(binned, observed=True)["PI"].median()
    energy_mean_aligned = energy_mean.reindex(counts.index, fill_value=0)
    binned_data = np.vstack((counts.values, energy_mean_aligned.values))
    return grid, binned_data


def create_windows(data):
    windows = []
    for start in range(0, data.shape[1] - window_size + 1, stride):
        windows.append(data[:, start : start + window_size])
    return np.stack(windows)

def train_model(device, model, windows):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    poisson_loss_fn = torch.nn.PoissonNLLLoss(log_input=False, full=True)
    mse_loss_fn = torch.nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(windows).float())
    train_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=False)
    
    model.train()
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward
            out, _ = model(batch)

            # Calcolo delle due componenti della loss
            loss_poisson = poisson_loss_fn(out, batch)
            loss_mse = mse_loss_fn(out, batch)

            loss = alpha * loss_poisson + beta * loss_mse

            # Regolarizzazione L2 leggera (weight decay manuale)
            reg_loss = sum(torch.sum(p ** 2) for p in model.parameters())
            loss += lambda_reg * reg_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        #print(f"Epoch {epoch+1}/{epochs} - Total Loss: {avg_loss:.6f}")
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        progress_text.text(f"Identifying noisy bins and reference part")
    return model


def reconstruct_curve(data, model, windows, device):
    windows = torch.from_numpy(windows).float().to(device)
    model.eval()
    with torch.no_grad():
        recon_windows,latent_out = model(windows)
        recon_windows = recon_windows.cpu().numpy()
        latent_out = latent_out.cpu().numpy()
    latent_out = latent_out[:,:,0]
    recon_sum = np.zeros_like(data)
    counts_overlap = np.zeros(data.shape[1])
    latent_sum = np.zeros((data.shape[1],latent_out.shape[1]))
    window_size = 16
    stride = 8
    start_indices = range(0, data.shape[1] - window_size + 1, stride)
    for i, start in enumerate(start_indices):
        recon_i = recon_windows[i].reshape((data.shape[0], window_size))
        recon_sum[:, start : start + window_size] += recon_i
        counts_overlap[start : start + window_size] += 1
        for j in range(start, start + window_size):
            latent_sum[j] += latent_out[i]

    counts_overlap[counts_overlap == 0] = 1
    reconstructed_curve = recon_sum / counts_overlap
    bin_embeddings = latent_sum / counts_overlap[:, None]
    reconstructed_curve = np.clip(reconstructed_curve, a_min=0, a_max=None)
    error = np.abs(reconstructed_curve - data).sum(axis=0)
    return error,bin_embeddings

def longest_stable_segment_from_error(error_array, binned_data, window=3, min_segment_len=5):
    n = len(error_array)
    counts_array = binned_data[0]
    error_array = (error_array - np.min(error_array)) / (np.max(error_array) - np.min(error_array))

    error_array = median_filter(error_array, size=3)

    median_count = np.median(counts_array)
    valid_bins = (counts_array > 0) & (counts_array < median_count)
    if not np.any(valid_bins):
        return np.array([], dtype=int)

    local_var = np.full(n, np.inf)
    for i in range(n):
        if not valid_bins[i]:
            continue
        start = max(0, i - window)
        end = min(n, i + window + 1)
        window_vals = error_array[start:end][valid_bins[start:end]]
        if len(window_vals) == 0:
            local_var[i] = np.inf
        else:
            local_var[i] = abs(error_array[i] - np.median(window_vals))

    finite_local_var = local_var[local_var != np.inf]
    if len(finite_local_var) == 0:
        return np.array([], dtype=int)
    threshold_relative = np.median(finite_local_var)
    stable = local_var <= threshold_relative

    segments = []
    start = None
    for i, val in enumerate(stable):
        if val:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_segment_len:
                segments.append((start, i))
            start = None
    if start is not None and n - start >= min_segment_len:
        segments.append((start, n))

    if not segments:
        return np.array([], dtype=int)

    max_len = max(end - start for start, end in segments)
    longest_segments = [(start, end) for start, end in segments if (end - start) == max_len]
    min_count_segment = min(longest_segments, key=lambda seg: np.median(counts_array[seg[0]:seg[1]]))
    best_start, best_end = min_count_segment
    return np.arange(best_start, best_end)

def find_noisy_bins(binned_data, good_part):
    counts = binned_data[0]

    gp_counts = counts[good_part]
    mu_gp = np.median(gp_counts)
    sigma_gp = np.std(gp_counts)
    threshold = mu_gp + 2 * sigma_gp
    
    noisy_bins = np.where(counts > threshold)[0]
    
    good_bins = np.where(counts <= threshold)[0]
    print(f"Number of noisy bins: {len(noisy_bins)}")
    print(f"Number of good bins: {len(good_bins)}")
    return good_bins, noisy_bins


def compute_reference_features(glowcurvedata, binned_data, grid, good_part):
    t_start = grid[good_part[0]]
    t_end = grid[good_part[-1] + 1]
    photons_good = glowcurvedata[(glowcurvedata["TIME"] >= t_start) & (glowcurvedata["TIME"] < t_end)]
    times = np.sort(photons_good["TIME"].values)
    energies = photons_good["PI"].values
    intertbinG = np.diff(times)
    target = np.median(intertbinG) if len(intertbinG) > 0 else 0
    goodElow = energies[(energies > 500) & (energies < 2000)]
    goodEhigh = energies[(energies > 2000) & (energies < 10000)]
    targetElow = np.median(goodElow) if len(goodElow) > 0 else 0
    targetEhigh = np.median(goodEhigh) if len(goodEhigh) > 0 else 0
    bin_counts = binned_data[0]
    median_good_count = np.median(bin_counts[good_part])
    sd_good_count = np.std(bin_counts[good_part])
    print(f"Reference features from good part:")
    print(f"  Median inter-photon time: {target:.4f} s")
    print(f"  Median low energy (500-2000 eV): {targetElow:.2f} eV")
    print(f"  Median high energy (2000-10000 eV): {targetEhigh:.2f} eV")
    print(f"  Median good bin count: {median_good_count:.2f}")
    print(f"  Std dev good bin count: {sd_good_count:.2f}")
    return target, targetElow, targetEhigh, median_good_count, sd_good_count


def rank_photons_by_similarity(target_time, targetE_low, targetE_high, noisy_photons, n):
    times = noisy_photons["TIME"].values
    energies = noisy_photons["PI"].values
    
    intertbinN = np.diff(times)
    if len(intertbinN) == 0:
        intertbinN = np.array([0.0])
    intertbinN = np.insert(intertbinN, 0, intertbinN[0])
    
    time_score = 1 / (1 + np.abs(intertbinN - target_time))
    energy_score = np.zeros_like(energies, dtype=float)
    energy_score[(energies >= targetE_low) & (energies <= targetE_high)] = 1
    
    penalty_mask = (energies >= 4000) & (energies <= 10000) 
    penalty = np.zeros_like(energies, dtype=float)
    penalty[penalty_mask] = -1.0
    combined_score = (weight_time) * time_score + (weight_energy) * energy_score + penalty
    combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min() + 1e-8)
    
    ranked_indices = np.argsort(combined_score)[::-1]
    top_indices = ranked_indices[:n]

    return noisy_photons.iloc[top_indices]

def clean_noisy_bins(
    obs_id,
    filename,
    nbins,
    glowcurvedata,
    binned_data,
    grid,
    noisy_bins,
    median_good_count,
    sd_mean_count,
    target,
    targetElow,
    targetEhigh,
    ):
    
    lower = int(median_good_count - sd_mean_count)
    upper = int(median_good_count + sd_mean_count)


    noisy_photons = []

    new_noisy_counts = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for i, bin_idx in tqdm(enumerate(noisy_bins), desc="Cleaning noisy bins", total=len(noisy_bins)):
        bin_idx = int(bin_idx)
        bin_count = binned_data[0, bin_idx]
        t_start = grid[bin_idx]
        t_end = grid[bin_idx + 1]

        bin_photons = glowcurvedata[(glowcurvedata["TIME"] >= t_start) & (glowcurvedata["TIME"] < t_end)]
        n_target = np.random.randint(lower, upper)
        new_noisy_counts.append(n_target)
        #print(f"Cleaning bin {bin_idx} with {bin_count} photons and {n_target} target photons")
        good_bin_photons = rank_photons_by_similarity(
            target,
            targetElow,
            targetEhigh,
            bin_photons,
            n_target,
        )
        
        all_bin_photons_idx = bin_photons.index.values
        noisy_photons_idx = np.setdiff1d(all_bin_photons_idx, good_bin_photons)
        noisy_photons.extend(noisy_photons_idx)
        progress = (i + 1) / len(noisy_bins)
        progress_bar.progress(progress)
        progress_text.text(f"Cleaning noisy bins")
        
    is_flair = np.zeros(len(glowcurvedata))
    is_flair[noisy_photons] = 1

    clean_curve = binned_data[0].copy()
    clean_curve[noisy_bins] = new_noisy_counts
    glowcurvedata["IS_FLAIR"] = is_flair
    t = Table.from_pandas(glowcurvedata)
    path = f"tmp/{filename}_{nbins}_DPC.fits"
    t.write(path, overwrite=True)
    tot = len(glowcurvedata)
    good_photons = len(glowcurvedata[glowcurvedata["IS_FLAIR"] == 0])
    flair_photons = len(glowcurvedata[glowcurvedata["IS_FLAIR"] == 1])
    fraction_removed = flair_photons / tot if tot > 0 else 0.0
    
    return path,clean_curve,tot,good_photons,fraction_removed


