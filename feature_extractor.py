import sklearn
import numpy as np
import pandas as pd

from scipy.fft import fft
from scipy.signal import welch

class FeatureExtractor:

    # Function to calculate Zero Crossing (ZC)
    def calculate_zc(self, signal):
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        return len(zero_crossings)

    # Function to calculate Slope Sign Changes (SSC)
    def calculate_ssc(self, signal):
        ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
        return ssc

    # Function to calculate Waveform Length (WL)
    def calculate_wl(self, signal):
        wl = np.sum(np.abs(np.diff(signal)))
        return wl

    # Fourier Transform (FT)
    def calculate_ft(self, signal):
        ft = fft(signal)
        return np.abs(ft)

    # Short-time Fourier Transform (SDFT)
    def calculate_sdft(self, signal, window_size):
        step_size = window_size // 2  # 50% overlap
        sdft_features = []
        for i in range(0, len(signal) - window_size, step_size):
            windowed_signal = signal[i:i+window_size]
            sdft = fft(windowed_signal)
            sdft_features.append(np.abs(sdft))
        return np.array(sdft_features)

    # Power Spectral Density (PSD)
    def calculate_psd(self, signal, fs=1000):
        freqs, psd = welch(signal, fs)
        return freqs, psd

    # RMS and Mean Absolute Value (MAV)
    def calculate_rms(self, signal):
        return np.sqrt(np.mean(np.square(signal)))

    def calculate_mav(self, signal):
        return np.mean(np.abs(signal))
    

    def extract_features(self, gesture_windows):
        features = []
        fs = 1000  # Sampling frequency
        channels = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8']

        for gesture_window in gesture_windows:
            channel_features = {}
            for channel in channels:
                signal = gesture_window[channel].values

                # Fourier Transform (FT) features
                ft_features = self.calculate_ft(signal)
                channel_features[channel + '_ft_mean'] = np.mean(ft_features)
                channel_features[channel + '_ft_max'] = np.max(ft_features)

                # Dominant Frequency (index of max frequency component)
                freqs, psd = self.calculate_psd(signal, fs)
                dominant_freq_idx = np.argmax(psd)
                dominant_frequency = freqs[dominant_freq_idx]
                channel_features[channel + '_dominant_frequency'] = dominant_frequency

                # SDFT Features
                window_size = 256  # Example window size
                sdft_features = self.calculate_sdft(signal, window_size)
                channel_features[channel + '_sdft_mean'] = np.mean(sdft_features)
                channel_features[channel + '_sdft_max'] = np.max(sdft_features)

                # Power Spectral Density (PSD) features
                channel_features[channel + '_psd_mean'] = np.mean(psd)
                channel_features[channel + '_psd_max'] = np.max(psd)
                channel_features[channel + '_total_power'] = np.sum(psd)

                # Spectral Band Power
                low_band_power = np.sum(psd[freqs < 30])  # Low band power (<30Hz)
                mid_band_power = np.sum(psd[(freqs >= 30) & (freqs < 60)])  # Mid band power (30-60Hz)
                high_band_power = np.sum(psd[freqs >= 60])  # High band power (>60Hz)
                channel_features[channel + '_low_band_power'] = low_band_power
                channel_features[channel + '_mid_band_power'] = mid_band_power
                channel_features[channel + '_high_band_power'] = high_band_power

                # Spectral Entropy
                normalized_psd = psd / np.sum(psd)
                spectral_entropy = -np.sum(normalized_psd * np.log(normalized_psd + 1e-12))
                channel_features[channel + '_spectral_entropy'] = spectral_entropy

                # Spectral Centroid and Bandwidth
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
                channel_features[channel + '_spectral_centroid'] = spectral_centroid
                channel_features[channel + '_spectral_bandwidth'] = spectral_bandwidth

                # Time-domain features
                channel_features[channel + '_zero_cross'] = self.calculate_zc(signal)
                channel_features[channel + '_slope_sign_c'] = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
                channel_features[channel + '_wavef_l'] = np.sum(np.abs(np.diff(signal)))
                channel_features[channel + '_rms'] = np.sqrt(np.mean(np.square(signal)))
                channel_features[channel + '_mav'] = np.mean(np.abs(signal))

                # Additional statistics (mean, min, max, variance)
                channel_features[channel + '_mean'] = np.mean(signal)
                channel_features[channel + '_min'] = np.min(signal)
                channel_features[channel + '_max'] = np.max(signal)
                channel_features[channel + '_variance'] = np.var(signal)

            # Add gesture class to the feature set
            channel_features['gesture'] = gesture_window['class'].values[0]
            features.append(channel_features)

        # Create a DataFrame from the extracted features
        feature_df = pd.DataFrame(features)
        return feature_df

            




