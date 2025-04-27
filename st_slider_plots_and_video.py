#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Скрипт для анализа данных двигателя с использованием Streamlit.
Позволяет загружать и анализировать видео, аудио и лог-файлы.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import plotly.graph_objects as go
import cv2
from scipy.signal import find_peaks
import tempfile
import os
from streamlit.runtime.scriptrunner import get_script_run_ctx

def main():
    # Настройка лимита загрузки файлов
    ctx = get_script_run_ctx()
    ctx.uploaded_file_mgr.max_upload_size = 500  # Лимит в МБ
    
    # Настройки страницы
    st.set_page_config(layout="wide")

    # Параметры анализа (вынесены в sidebar)
    with st.sidebar:
        st.header("Параметры анализа")
        x1 = st.number_input("x1 ROI", value=630)
        y1 = st.number_input("y1 ROI", value=300)
        x2 = st.number_input("x2 ROI", value=930)
        y2 = st.number_input("y2 ROI", value=600)
        prominence = st.slider("Проминенс для детекции минимумов", 0.1, 5.0, 1.0)

    # Интерфейс
    st.title("Анализ данных двигателя")
    uploaded_video = st.sidebar.file_uploader("Видео (MOV)", type=["mov"])
    uploaded_audio = st.sidebar.file_uploader("Аудио (WAV)", type=["wav"])
    uploaded_log = st.sidebar.file_uploader("Лог (TXT)", type=["txt"])

    if uploaded_video and uploaded_audio and uploaded_log:
        data = load_data(uploaded_video, uploaded_audio, uploaded_log, x1, y1, x2, y2, prominence)
        y, sr, duration = data['audio']
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        # Основной интерфейс
        current_time = st.slider(
            "Текущее время (сек)",
            0.0, float(duration), 0.0, 0.01,
            format="%.2f", key="time_slider"
        )

        # Организация колонок
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.subheader("Видео")
            st.video(data['video_path'], start_time=int(current_time))

            fig, ax = plt.subplots(figsize=(10, 2))
            start_sample = int(max(0, (current_time - 2.5) * sr))
            end_sample = int(min(len(y), (current_time + 2.5) * sr))
            librosa.display.waveshow(y[start_sample:end_sample], sr=sr, ax=ax)
            ax.axvline(x=current_time - (start_sample/sr), color='r')
            st.pyplot(fig)

        with col2:
            st.subheader("Спектрограмма")
            fig = go.Figure()
            times_spec = librosa.times_like(D, sr=sr)
            mask = (times_spec >= max(0, current_time-2.5)) & (times_spec <= min(duration, current_time+2.5))
            fig.add_trace(go.Heatmap(
                z=D[:, mask],
                x=times_spec[mask],
                y=librosa.fft_frequencies(sr=sr),
                colorscale='magma'
            ))
            fig.add_vline(x=current_time, line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            st.subheader("Интенсивность")
            fig = go.Figure(go.Scatter(x=data['log_data'][0], y=data['log_data'][1]))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("Счетчик")
            fig = go.Figure(go.Scatter(x=data['log_data'][0], y=data['log_data'][2]))
            st.plotly_chart(fig, use_container_width=True)

        with col5:
            st.subheader("Видеоанализ")
            fig = go.Figure(go.Scatter(
                x=data['video_analysis'][0], 
                y=data['video_analysis'][1]
            ))
            for t in data['video_analysis'][2]:
                fig.add_vline(x=t, line_color="blue", line_dash="dot", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # Очистка
        for f in data['temp_files']:
            try: os.unlink(f)
            except: pass
    else:
        st.warning("Загрузите все файлы для анализа")

@st.cache_data
def load_data(uploaded_video, uploaded_audio, uploaded_log, x1, y1, x2, y2, prominence):
    """
    Загружает и обрабатывает данные из загруженных файлов.
    
    Args:
        uploaded_video: Загруженный видеофайл
        uploaded_audio: Загруженный аудиофайл
        uploaded_log: Загруженный лог-файл
        x1, y1, x2, y2: Координаты ROI для анализа видео
        prominence: Параметр для детекции минимумов
        
    Returns:
        dict: Словарь с обработанными данными
    """
    # Сохраняем файлы во временные файлы
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        audio_path = tmp_audio.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_log:
        tmp_log.write(uploaded_log.read())
        log_path = tmp_log.name

    # Загрузка и обработка данных
    y, sr = librosa.load(audio_path)
    duration = len(y)/sr

    # Обработка лог-файла
    with open(log_path, 'r') as file:
        data = [line.split() for line in file.read().strip().split('\n')]
        
        base_times = [int(d[2]) for d in data]
        time_counts = {}
        for t in base_times:
            time_counts[t] = time_counts.get(t, 0) + 1
            
        times, intensities, cnt = [], [], []
        for i, d in enumerate(data):
            t = base_times[i]
            if time_counts[t] > 1:
                curr_count = sum(1 for x in base_times[:i] if x == t)
                offset = curr_count * (1.0 / time_counts[t])
                times.append(t + offset)
            else:
                times.append(t)
            intensities.append(float(d[0]))
            cnt.append(int(d[1]))
    
    times = [t * 6 / 1000 for t in times]  # Конвертация времени

    # Анализ видео
    cap = cv2.VideoCapture(video_path)
    frame_timestamps = []
    while cap.isOpened():
        ret, _ = cap.read()
        if not ret: break
        frame_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    synced_results = []
    for frame_idx in range(len(frame_timestamps)):
        ret, frame = cap.read()
        if not ret or frame is None: continue
        
        try:
            red_intensity = np.mean(frame[y1:y2, x1:x2, 2])
        except:
            red_intensity = 0
        
        synced_results.append({
            'frame_time': frame_timestamps[frame_idx],
            'red': red_intensity
        })
    cap.release()

    # Дополнительная обработка
    times_sec = np.array([x['frame_time'] for x in synced_results])
    red = np.array([x['red'] for x in synced_results])
    time_offset = times_sec[0]
    times_sec_corrected = times_sec - time_offset

    minima_indices, _ = find_peaks(-red, prominence=prominence)
    minima_times = times_sec_corrected[minima_indices]

    return {
        'audio': (y, sr, duration),
        'video_path': video_path,
        'log_data': (times, intensities, cnt),
        'video_analysis': (times_sec_corrected, red, minima_times),
        'temp_files': (video_path, audio_path, log_path)
    }

if __name__ == "__main__":
    main()