#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для анализа аудиофайла.
Загружает аудиофайл, вычисляет его длительность и количество сэмплов.
"""

import librosa
import argparse


def analyze_audio(file_path):
    """
    Анализирует аудиофайл и возвращает его длительность и количество сэмплов.
    
    Args:
        file_path (str): Путь к аудиофайлу
        
    Returns:
        tuple: (длительность в секундах, количество сэмплов)
    """
    audio, sr = librosa.load(file_path, sr=None)  # sr - частота дискретизации
    duration = len(audio) / sr  # Длительность в секундах
    total_samples = len(audio)  # Искомое количество измерений
    
    return duration, total_samples


def main():
    parser = argparse.ArgumentParser(description='Анализ аудиофайла')
    parser.add_argument('--file', required=True, help='Путь к аудиофайлу для анализа')
    args = parser.parse_args()
    
    duration, total_samples = analyze_audio(args.file)
    
    print(f"Длительность аудио: {duration} секунд")
    print(f"Количество сэмплов: {total_samples}")


if __name__ == "__main__":
    main()