import streamlit as st
from DeepPhotonCleaner import (
    read_fits, bin_data, identify_device, create_windows,
    train_model, reconstruct_curve, longest_stable_segment_from_error,
    find_noisy_bins, compute_reference_features, clean_noisy_bins
)
from model import MultichannelAutoencoder
from plot_utils import plot_curve, plot_results, plot_spectra
from interactive_plot import plot_curve_plotly, plot_results_plotly, plot_spectra_plotly
import os
from io import BytesIO
import zipfile
import shutil

# --- Funzioni ---
def download_temp_folder(tmp_dir="tmp"):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(tmp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)
    buffer.seek(0)
    return buffer

def reset_tmp_folder(tmp_dir="tmp"):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

# --- Session state ---
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = "tmp"
    os.makedirs(st.session_state.tmp_dir, exist_ok=True)

# --- Page setup ---
st.set_page_config(page_title="DeepPhotonCleaner", layout="wide")
st.image("logo.png", width=200)
st.title("🔭 DeepPhotonCleaner")
st.subheader("Deep learning to clean light-curves, separating signals from noise due to solar flares.")

# --- Observation details ---
st.markdown("### 🧩 Observation Details")
col1, col2, col3 = st.columns(3)
with col1:
    observation_id = st.text_input("Observation ID",key = "observation_id")
with col2:
    camera = st.selectbox("Camera", ["mos1", "mos1", "pn"],key = "camera")
with col3:
    nbins = st.selectbox("Number of bins", [2**13, 2**14, 2**15, 2**16], index=0, key = "nbins")

# --- Show FITS uploaders ---
if observation_id and camera and nbins:
    st.markdown("### 📁 Upload FITS Files")
    col_a, col_b = st.columns(2)
    with col_a:
        fits_noisy = st.file_uploader("Upload noisy `.fits` file", type=["fits"], key = "fits_noisy")
    with col_b:
        fits_clean_sas = st.file_uploader("Upload XMM-SAS cleaned `.fits` file", type=["fits"], key = "fits_clean_sas")

    if fits_noisy and fits_clean_sas:
        st.success("✅ Both FITS files uploaded successfully!")

        noisy_path = os.path.join(st.session_state.tmp_dir, fits_noisy.name)
        fits_noisy.seek(0)
        with open(noisy_path, "wb") as f:
            f.write(fits_noisy.read())

        clean_sas_path = os.path.join(st.session_state.tmp_dir, fits_clean_sas.name)
        fits_clean_sas.seek(0)
        with open(clean_sas_path, "wb") as f:
            f.write(fits_clean_sas.read())

        glowcurvedata = read_fits(noisy_path)
        grid, binneddata = bin_data(glowcurvedata, nbins)
        noisycounts = binneddata[0]
        filename = fits_noisy.name.split(".fits")[0]

        # Plot iniziale
        plot_curve(observation_id, filename, nbins, grid, [noisycounts])
        plot_curve_plotly(observation_id, filename, nbins, grid, [noisycounts])

        # --- Cleaning ---
        if st.button("🚀 Run Cleaning") or st.session_state.results_ready:
            if not st.session_state.results_ready:
                device, devname = identify_device()
                windows = create_windows(binneddata)
                model = MultichannelAutoencoder().to(device)
                model = train_model(device, model, windows)
                error, embs = reconstruct_curve(binneddata, model, windows, device)
                good_part = longest_stable_segment_from_error(error, binneddata, window=3)
                startG, endG = good_part[0], good_part[-1]
                target, targetElow, targetEhigh, median_good_count, sd_good_count = compute_reference_features(
                    glowcurvedata, binneddata, grid, good_part
                )
                goodbins, noisybins = find_noisy_bins(binneddata, good_part)
                
                st.markdown("### 📦 Reference Features & Good Part Info")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start Good Part", f"{startG}")
                    st.metric("Target Low energy", f"{targetElow:.4f}")
                    st.metric("Median photons", f"{median_good_count:.4f}")
                with col2:
                    st.metric("End Good Part", f"{endG}")
                    st.metric("Target High energy", f"{targetEhigh:.4f}")
                    st.metric("Std dev photons", f"{sd_good_count:.4f}")
                with col3:
                    st.metric("Target Inter-Photon Time", f"{target:.4f}")
                    st.metric("Noisy Bins", f"{len(noisybins)}")
                    st.metric("Clean Bins", f"{len(goodbins)}")

                clean_fits, clean_curve, tot, good_photons, fraction_removed = clean_noisy_bins(
                    observation_id, filename, nbins, glowcurvedata, binneddata, grid, noisybins,
                    median_good_count, sd_good_count, target, targetElow, targetEhigh
                )
                
                st.markdown("### ✅ Cleaning Results")
                st.metric("Total photons", f"{tot}")
                st.metric("Good photons", f"{good_photons}")
                st.metric("Fraction removed", f"{fraction_removed:.4f} %")

                plot_results_plotly(observation_id, filename, nbins, grid, binneddata, clean_curve, good_part)
                plot_spectra_plotly(observation_id, filename, nbins, noisy_path, clean_sas_path, clean_fits)

                plot_results(observation_id, filename, nbins, grid, binneddata, clean_curve, good_part)
                plot_spectra(observation_id, filename, nbins, noisy_path, clean_sas_path, clean_fits)
                # Salva nello stato
                st.session_state.results_ready = True

            # --- Download ---
            zip_buffer = download_temp_folder(st.session_state.tmp_dir)
            st.download_button(
                label="📥 Download All Results",
                data=zip_buffer,
                file_name=f"obs_{observation_id}_{nbins}.zip",
                mime="application/zip"
            )

            # --- Reset ---
            if st.button("🧹 Reset All"):
                # Cancella tutti i file temporanei
                reset_tmp_folder(st.session_state.tmp_dir)
                keys_to_clear =[
                    "results_ready",
                    "observation_id",
                    "camera",
                    "nbins",
                    "fits_noisy",
                    "fits_clean_sas",
                ]
                
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]

                # Ricarica la pagina
                st.rerun()

                # Resetta tutti i valori salvati nello session_state

else:
    st.info("Please fill in all observation details to upload FITS files.")