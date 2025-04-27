"""
Monitor Drill Analysis Tool

This script processes video and audio data from drill monitoring equipment to analyze 
rotation patterns and generate visualizations. It combines data from multiple sources:

- Video footage of the drill in operation
- Raw audio recordings from sensors
- Log files with rotation counts

The script produces a comprehensive visualization showing:
1. Raw sensor data
2. Audio spectrograms from both sources 
3. Video-based intensity analysis
4. Frequency calculations

Author: Unknown
Created: Unknown
"""

# Standard library imports
import argparse
import os
import subprocess
import time
from datetime import datetime, timedelta

# Third party imports
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Local imports
from configs import config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process drill monitoring data')
    parser.add_argument('--log-number', '-l', 
                       type=int, 
                       required=True,
                       help='Log number to process')
    return parser.parse_args()


def get_video_start_unix(video_path):
    """
    Get video start time in Unix timestamp.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        float: Unix timestamp of video start
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format_tags=creation_time",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    creation_time = subprocess.check_output(cmd).decode("utf-8").strip()
    
    dt = datetime.strptime(creation_time, "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=3)
    print(f"Начало видео (datetime +3): {dt}")
    return time.mktime(dt.timetuple())


def process_video_frames(cap, x1, y1, x2, y2):
    """
    Process video frames to extract timestamps and red channel intensity.
    
    Args:
        cap: OpenCV video capture object
        x1, y1, x2, y2: ROI coordinates
        
    Returns:
        tuple: List of frame timestamps and list of analysis results
    """
    frame_timestamps = []
    synced_results = []
    
    # Collect frame timestamps
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process each frame
    for frame_idx in range(len(frame_timestamps)):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Пропущен кадр {frame_idx}")
            continue
        
        frame_time = frame_timestamps[frame_idx]
        
        try:
            red_intensity = np.mean(frame[y1:y2, x1:x2, 2])
        except Exception as e:
            print(f"Ошибка обработки кадра {frame_idx}: {e}")
            red_intensity = 0
        
        synced_results.append({
            'frame_time': frame_time,
            'red': red_intensity
        })
        
    return frame_timestamps, synced_results


def read_log_data(log_path):
    """
    Read and parse log file data.
    
    Args:
        log_path: Path to log file
        
    Returns:
        tuple: Audio start time, timestamps and counts
    """
    with open(log_path, 'r') as file:
        data = file.read().strip().split('\n')
        data = [d.split(',') for d in data[1:]]
        audio_start_unix = float(data[0][0])
        times = [float(d[0]) for d in data]
        cnt = [int(d[1]) for d in data]
    
    print(f"Начало audio (datetime +0): {datetime.fromtimestamp(audio_start_unix)}")
    print(f"Начало audio (Unix): {audio_start_unix}")
    
    return audio_start_unix, times, cnt


def process_audio(raw_audio_path, audio_path, stft_params):
    """
    Process both raw and video audio files.
    
    Args:
        raw_audio_path: Path to raw audio file
        audio_path: Path to video audio file
        stft_params: STFT processing parameters
        
    Returns:
        tuple: Raw audio data and video audio data
    """
    # Process raw audio
    y, sr = librosa.load(raw_audio_path, sr=None)
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, **stft_params)),
        ref=np.max,
        top_db=80
    )

    # Process video audio
    y2, sr2 = librosa.load(audio_path, sr=None)
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
    
    return (y, sr, D), (y2, sr2, D2)


def calculate_frequency(times_sec, minima_times, window=1):
    """
    Calculate frequency of minima occurrences.
    
    Args:
        times_sec: Array of timestamps
        minima_times: Array of minima timestamps
        window: Window size in seconds
        
    Returns:
        tuple: Arrays of times and counts
    """
    times_for_counts = []
    counts = []

    for t in np.arange(0, times_sec[-1], 0.1):
        window_start = max(t - window/2, 0)
        window_end = min(t + window/2, times_sec[-1])
        
        window_minima = minima_times[(minima_times >= window_start) & 
                                   (minima_times <= window_end)]
        count = len(window_minima)
        
        times_for_counts.append(t)
        counts.append(count / window)

    return np.array(times_for_counts), np.array(counts)


def main():
    """Main execution function."""
    # Parse arguments and get configuration
    args = parse_arguments()
    log_number = args.log_number
    video_number = config.log_to_video[log_number]

    # Set up file paths
    log_path = os.path.join(config.basedir, 'data', 'logs', f'обороты_{log_number}.txt')
    video_path = os.path.join(config.basedir, 'data', 'videos', f'IMG_{video_number}.MOV')
    raw_audio_path = os.path.join(config.basedir, 'data', 'audios', f'audio_{log_number}.wav')
    audio_path = os.path.join(config.basedir, 'data', 'audios', f'audio_{video_number}_stereo.wav')

    # ROI coordinates for video processing
    x1, y1, x2, y2 = 630, 200, 930, 500

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видеофайл!")

    # Process video frames
    frame_timestamps, synced_results = process_video_frames(cap, x1, y1, x2, y2)
    cap.release()

    # Read log data
    audio_start_unix, times, cnt = read_log_data(log_path)

    # Get video timing information
    video_start_unix = get_video_start_unix(video_path)
    print(f"Начало видео (Unix): {video_start_unix}")

    start_timestamp = min(video_start_unix, audio_start_unix)
    video_start_unix += config.video_audio_ms_deltas[log_number]

    # Process video data
    prominence = config.prominences[log_number]
    times_sec = np.array([x['frame_time'] for x in synced_results])
    red = np.array([x['red'] for x in synced_results])
    time_offset = times_sec[0]
    times_sec_corrected = times_sec - time_offset

    # Find minima
    minima_indices, _ = find_peaks(-red, prominence=prominence)
    minima_times = times_sec_corrected[minima_indices]

    # Audio processing parameters
    stft_params = {
        'n_fft': 512,
        'hop_length': 128,
        'win_length': 512,
        'window': 'hann'
    }

    # Process audio files
    raw_audio_data, video_audio_data = process_audio(raw_audio_path, audio_path, stft_params)
    y, sr, D = raw_audio_data
    y2, sr2, D2 = video_audio_data

    # Create visualization
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(150, 20), 
                                            gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

    # Define plot boundaries
    x_min = min(audio_start_unix - start_timestamp, video_start_unix - start_timestamp)
    x_max = max(
        len(y)/sr + audio_start_unix - start_timestamp,
        len(y2)/sr2 + video_start_unix - start_timestamp
    )

    # Plot 1: Raw audio spectrogram
    im1 = ax2.imshow(D, aspect='auto', origin='lower', cmap='magma',
                   extent=[audio_start_unix - start_timestamp, 
                          len(y)/sr + audio_start_unix - start_timestamp, 
                          0, sr/2], 
                   vmin=-80, vmax=0)
    ax2.set_ylabel('Частота (Гц)')
    ax2.set_ylim(0, 2000)
    ax2.set_xlim(x_min, x_max)
    ax2.set_title('Спектрограмма аудио с датчика')

    # Plot 2: Sensor data
    times = [t - start_timestamp for t in times]
    ax1.plot(times, cnt, 'r-', linewidth=1.5)
    ax1.set_ylabel('Обороты по аудио')
    ax1.set_xlim(x_min, x_max)
    ax1.grid(True)
    ax1.set_title('Данные с датчика')

    # Plot 3: Processed audio spectrogram
    im2 = ax3.imshow(D2, aspect='auto', origin='lower', cmap='magma',
                   extent=[video_start_unix - start_timestamp, 
                          len(y2)/sr2 + video_start_unix - start_timestamp, 
                          0, sr2/2], 
                   vmin=-80, vmax=0)
    ax3.set_ylabel('Частота (Гц)')
    ax3.set_ylim(0, 2000)
    ax3.set_xlim(x_min, x_max)
    ax3.set_title('Спектрограмма аудио из видео')

    # Plot 4: Intensity graph with minima
    times_sec = np.array([x['frame_time'] + video_start_unix - start_timestamp for x in synced_results])
    red = np.array([x['red'] for x in synced_results])

    minima_indices, _ = find_peaks(-red, prominence=prominence)
    minima_times = times_sec[minima_indices]

    # Draw main plot and vertical lines
    ax4.plot(times_sec, red, 'r-', linewidth=1.5)
    for t in minima_times:
        for ax in [ax2, ax3, ax4]:
            ax.axvline(x=t, color='blue', linestyle='--', alpha=0.5, linewidth=0.3)

    ax4.set_xlabel('Время (сек)')
    ax4.set_ylabel('Интенсивность по видео')
    ax4.set_xlim(x_min, x_max)
    ax4.grid(True)
    ax4.set_title('График интенсивности с минимумами')

    # Calculate and plot frequency
    times_for_counts, counts = calculate_frequency(times_sec, minima_times)
    ax4.plot(times_for_counts, counts, 'g-', 
             linewidth=1, alpha=0.7, label='Количество минимумов')
    ax4.legend(loc='upper left')

    # Set x-axis ticks
    tick_step = 0.1
    xticks = np.arange(times_sec[0], times_sec[-1], tick_step)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=90)

    # Save plot
    plt.savefig(os.path.join(config.basedir, 'artefacts', 'plots', 
                            f'spectrogram_from_both_video_{video_number}_(log{log_number}).png'), 
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()