import plotly.graph_objects as go
import streamlit as st
import numpy as np
from astropy.io import fits
import pandas as pd

def convert_endian(df):
    for col in df.columns:
        dtype = df[col].dtype
        if dtype.byteorder == ">" and np.issubdtype(dtype, np.number):
            swapped = df[col].values.byteswap()
            df[col] = swapped.view(swapped.dtype.newbyteorder()).copy()
    return df


def plot_curve_plotly(obs_id, filename, nbins, grid, binned_data):
    bin_centers = (grid[:-1] + grid[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_centers, y=binned_data[0],
        mode='lines+markers',
        name='Original Curve',
        line=dict(color='black')
    ))
    fig.update_layout(
        title=f"Observation {obs_id} - {filename}, Noisy Curve N.Bins={nbins}",
        xaxis_title="Time",
        yaxis_title="Counts",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    


def plot_results_plotly(obs_id, filename, nbins, grid, data, cleaned_curve, good_bins):
    bin_centers = (grid[:-1] + grid[1:]) / 2

    fig = go.Figure()

    # Original curve
    fig.add_trace(go.Scatter(
        x=bin_centers, y=data[0],
        mode='lines+markers',
        name='Original Curve',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))

    # Clean curve
    fig.add_trace(go.Scatter(
        x=bin_centers, y=cleaned_curve,
        mode='lines+markers',
        name='Clean Curve (DPC)',
        line=dict(color='green', dash='dash'),
        marker=dict(size=6)
    ))

    # Highlight good bins region
    if good_bins is not None and len(good_bins) > 0:
        t_start = float(bin_centers[min(good_bins)])
        t_end = float(bin_centers[max(good_bins)])
        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor="orange", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Reference Part", annotation_position="top left"
        )

    fig.update_layout(
        title=f"DeepPhotonCleaner - Observation {obs_id} {filename}, N.Bins={nbins}",
        xaxis_title="Time",
        yaxis_title="Counts",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    

def plot_spectra_plotly(obs_id, filename, nbins, ori_path, sas_path, clean_path):
    bin_width = 0.1
    bins = np.arange(0.3, 15.0 + bin_width, bin_width)

    # --- Raw curve ---
    with fits.open(ori_path) as hdul:
        data = pd.DataFrame(hdul[1].data)
    energies = np.array(data["PI"]) / 1000
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, edges = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm = counts / (n_photons * bin_width)

    # --- SAS cleaned ---
    with fits.open(sas_path) as hdul:
        data = pd.DataFrame(hdul[1].data)
    data = convert_endian(data)
    energies = np.array(data["PI"]) / 1000
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, _ = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm_allevc = counts / (n_photons * bin_width)

    # --- DPC cleaned ---
    with fits.open(clean_path) as hdul:
        data = pd.DataFrame(hdul[1].data)
    data = convert_endian(data)
    data = data[data["IS_FLAIR"] == 0]
    energies = np.array(data["PI"]) / 1000
    energies_sel = energies[(energies >= 0.3) & (energies <= 15.0)]
    counts, _ = np.histogram(energies_sel, bins=bins)
    n_photons = len(energies_sel)
    counts_norm_dpc = counts / (n_photons * bin_width)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # --- Plotly figure ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_centers, y=counts_norm,
        mode='lines+markers', name='Raw', line=dict(color='black')
    ))
    fig.add_trace(go.Scatter(
        x=bin_centers, y=counts_norm_allevc,
        mode='lines+markers', name='SAS', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=bin_centers, y=counts_norm_dpc,
        mode='lines+markers', name=f'fDPC (N.Bins={nbins})', line=dict(color='green')
    ))

    fig.update_layout(
        title=f"Spectra Comparison for Observation {obs_id} - {filename}",
        xaxis_title="Energy (keV)",
        yaxis_title="Normalized Counts (per keV)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)