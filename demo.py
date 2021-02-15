from enum import Enum
from io import BytesIO, StringIO
from typing import Union

import pandas as pd
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

import audioread
import io
import datetime


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

FILE_TYPES = ["mp3", "ogg", "flac", "m4a"]


def compute_melspec(S_full):
    return librosa.amplitude_to_db(S_full, ref=np.max)


# @st.cache
def plot_spectrogram(ax, melspec, sr, slice_min, slice_max):
    # Plot a slice of the spectrogram
    idx = slice(*librosa.time_to_frames([slice_min, slice_max], sr=sr))
    librosa.display.specshow(melspec[:,idx], y_axis='log', x_axis='time', sr=sr, ax=ax)
                        
            
def build_filter(S_full, sr):
    # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)

    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                margin_i * (S_full - S_filter),
                                power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                margin_v * S_filter,
                                power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    

    return compute_melspec(S_foreground), compute_melspec(S_background)


def plot_separated(ax, melspec, melspec_background, melspec_foreground, sr, slice_min, slice_max):
    # Plot a slice of the spectrogram
    idx = slice(*librosa.time_to_frames([slice_min, slice_max], sr=sr))
    
    librosa.display.specshow(melspec[:,idx], y_axis='log', x_axis='time', sr=sr, ax=ax[0])
    ax[0].set(title='Full spectrum')
    ax[0].label_outer()

    librosa.display.specshow(melspec_background[:,idx], y_axis='log', x_axis='time', sr=sr, ax=ax[1])
    ax[1].set(title='Background')
    ax[1].label_outer()

    librosa.display.specshow(melspec_foreground[:,idx], y_axis='log', x_axis='time', sr=sr, ax=ax[2])
    ax[2].set(title='Foreground')


# @st.cache
def load_file(audio_file):
    content, filename = audio_file.read(), '/Users/athon/Documents/' + audio_file.name
    return librosa.load(filename, duration=120)


# @st.cache
def generate_spectrograms(y, sr):
    S_full, phase = librosa.magphase(librosa.stft(y))

    melspec = compute_melspec(S_full)
    melspec_background, melspec_foreground = (compute_melspec(S) for S in (build_filter(S_full, sr)))
    
    return melspec, melspec_background, melspec_foreground


def main():
    """Run this function to display the Streamlit app"""
    # st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)

    col1, col2 = st.beta_columns(2)

    # Audio upload
    audio_file = col1.file_uploader("Upload file", type=FILE_TYPES)
    show_file = col1.empty()
    if not audio_file:
        show_file.info("Please upload a file of type: " + ", ".join(FILE_TYPES))
        return

    show_file.audio(audio_file, format='audio/mp3')
    y, sr = load_file(audio_file)

    slice_min, slice_max = col1.slider(f"Window", 0, 120, (0,120))
    col1.write(f"{str(datetime.timedelta(seconds=slice_min))} -- {str(datetime.timedelta(seconds=slice_max))}")


    # Generate spectrograms
    melspec, melspec_background, melspec_foreground = generate_spectrograms(y, sr)

    # Plot Spectrogram
    spectrogram = col2.empty()
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=[12,12])
    # fig, ax = plt.subplots(figsize=[16,8])
    
    # plot_spectrogram(ax, melspec, sr, slice_min, slice_max)
    plot_separated(ax, melspec, melspec_background, melspec_foreground, sr, slice_min, slice_max)
    spectrogram.pyplot(fig)

    # Process audio
    

    # Plot embeddings



main()