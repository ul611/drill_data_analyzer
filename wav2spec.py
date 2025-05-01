import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import argparse
from configs.config import basedir

def generate_spectrogram(video_number, audio_path):
    
    # Используем basedir для формирования путей
    if audio_path is None:
        audio_path = os.path.join(os.path.dirname(basedir), 'data/audios', f'audio_{video_number}_stereo.wav')
    if video_number is None:
        video_number = os.path.basename(audio_path)
    # Проверяем существование аудио файла
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудио файл не найден: {audio_path}")
    
    # Загружаем аудио
    y, sr = librosa.load(audio_path, sr=None)
    print("Длина аудио файла:", len(y))
    
    # Создаем спектрограмму
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, window='hamming')), ref=np.max)

    duration = len(y) / sr
    figsize_x = int(duration * 2.5)
    
    # Создаем и настраиваем график

    fig, ax1 = plt.subplots(1, 1, figsize=(figsize_x, 10), sharex=True, gridspec_kw={'hspace': 0.05})
    im = ax1.imshow(D, aspect='auto', origin='lower', cmap='magma',
                   extent=[0, len(y)/sr, 0, sr/2], vmin=-80, vmax=0)
    ax1.set_ylim(0, 8000)
    # Добавляем подписи осей
    ax1.set_ylabel('Частота (Гц)')
    ax1.set_xlabel('Время (сек)')
    ax1.set_title('Спектрограмма аудио')
    
    # Устанавливаем метки по оси X с шагом 0.1 секунды и поворотом на 90 градусов
    tick_step = 0.1
    xticks = np.arange(0, len(y)/sr, tick_step)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=90)
    
    # Добавляем сетку для лучшей читаемости
    ax1.grid(True, alpha=0.3)
    
    # Путь для сохранения спектрограммы в папку с артефактами
    # output_path = os.path.join(basedir, 'artefacts/plots/plot_for_video_to_markup', f'spec_for_video_{video_number}.png')
    output_path = audio_path.replace('.wav', '.png')
    
    # Сохраняем спектрограмму
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Спектрограмма сохранена: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Создание спектрограммы для аудио файла')
    parser.add_argument('--video_number', '-v', type=int, help='Номер видео для обработки', default=None)
    parser.add_argument('--audio_path', '-a', type=str, help='Путь к аудио файлу', default=None)
    
    args = parser.parse_args()
    generate_spectrogram(args.video_number, args.audio_path)