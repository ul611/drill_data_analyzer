# 📊 Drill Data Analyzer

A toolset for analyzing real-world drill operation data, including video and audio processing.

## 📋 Contents

- [Overview](#overview)
- [Installation](#installation)
- [Tools](#tools)
- [Tech Details](#tech-details)
- [Project Structure](#project-structure)

## 📝 Overview

This project provides tools for:
- Analyzing sensor data and video recordings of drill operations
- Extracting and processing audio from video files
- Visualizing data through graphs and spectrograms
- Monitoring drill performance under various conditions

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/username/drill_data_analyzer.git
cd drill_data_analyzer

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Tools

### 1. Drill Video Analyzer (monitor_drill.py)

```bash
python monitor_drill.py --log-number LOG_NUMBER
```
Analyzes drill operation videos with corresponding log data.

### 2. Audio Extractor (mov2wav.py)

```bash
python mov2wav.py --video VIDEO_PATH --audio AUDIO_PATH
```
Extracts audio tracks from video files for analysis.

### 3. Data Visualization (video_and_spec_with_frames.py, st_slider_plots_and_video.py)

```bash
streamlit run st_slider_plots_and_video.py
```
Interactive data and video visualization using Streamlit.

## 🔬 Tech Details

Key technologies used:
- **OpenCV** for video processing
- **FFmpeg** for media handling
- **Librosa** for audio analysis
- **NumPy/SciPy** for data processing
- **Matplotlib** for visualization
- **Streamlit** for web interface

## 📁 Project Structure

```
drill_data_analyzer/
├── monitor_drill.py          # Drill video analyzer
├── mov2wav.py                # Audio extraction
├── video_and_spec_with_frames.py  # Video frame processing
├── st_slider_plots_and_video.py   # Interactive plots
├── plot_existing_audio.py    # Audio visualization
├── configs/                  # Configuration files
│   └── config.py             # Main settings
└── requirements.txt          # Dependencies
```

## 📊 Data Requirements

The project works with:
- Drill operation videos
- Sensor data logs
- Audio files (can be extracted from videos)

Recommended data structure:
```
data/
├── videos/      # Video files (*.MOV)
├── audios/      # Audio files (*.wav)
└── logs/        # Sensor logs
```
