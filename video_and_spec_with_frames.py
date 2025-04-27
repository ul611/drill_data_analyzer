import argparse
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
import librosa.display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import gc


parser = argparse.ArgumentParser(description='Process video and generate spectrograms')
parser.add_argument('--video-number', '-v', type=int, required=True, help='Video number to process')
args = parser.parse_args()
video_number = args.video_number

video_path = f"/home/ul/plot_drill/data/videos/IMG_{video_number}.MOV"
audio_path = f'/home/ul/plot_drill/data/audios/audio_{video_number}_stereo.wav'

# Настройки
INPUT_VIDEO = video_path
OUTPUT_VIDEO = f"video_with_frames_and_spec_{video_number}.mp4"
AUDIO_PATH = audio_path
FRAMES_DIR =  f"output_frames_{video_number}"
WINDOW_SECONDS = 10.0  # Измените на нужную ширину окна
GRAPH_HEIGHT = 600    # Высота графика в пикселях

# Освобождение ресурсов Matplotlib
def clean_plt():
    plt.close('all')
    gc.collect()

# Создаем папку для кадров
os.makedirs(FRAMES_DIR, exist_ok=True)

# Инициализация видео
ratio = 0.7
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
# Check if video is vertical (height > width)
if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
    # For vertical video, swap width and height and rotate later
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio)
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
else:
    # For horizontal video, keep original dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Загрузка аудио и данных
print("Загрузка аудио...")
y, sr = librosa.load(AUDIO_PATH, sr=None)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y, window='hamming')), ref=np.max)
times_spec = librosa.times_like(D, sr=sr)

# Создаем фигуру один раз
print("Создание фигуры...")
plt.switch_backend('Agg')
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10), 
                       sharex=True, gridspec_kw={'hspace': 0.05}, dpi=100)

# Настраиваем оси
ax1.set_ylabel('Частота (Гц)')
ax1.set_ylim(0, 8000)
divider = make_axes_locatable(ax1)

# Создаем пустой imshow для последующего обновления
im = ax1.imshow(np.zeros((D.shape[0], 1)), aspect='auto', origin='lower',
               cmap='magma', extent=[0, 1, 0, sr/2],
               vmin=-80, vmax=0)

# Обработка каждого кадра
print("Обработка кадров...")
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Поворачиваем кадр на 90 градусов влево для вертикального видео
    if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    current_time = frame_idx / fps
    
    # Вычисляем границы окна так, чтобы текущее время было всегда в центре
    half_window = WINDOW_SECONDS / 2
    start_time = max(0, min(current_time - half_window, total_frames / fps - WINDOW_SECONDS))
    end_time = min(max(WINDOW_SECONDS, current_time + half_window), total_frames / fps)
    
    # Добавляем номер кадра в левый нижний угол
    frame_text = f"Frame: {frame_idx}"
    (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, height - text_height - 30), (10 + text_width + 20, height - 10), (255, 255, 255), -1)
    cv2.putText(frame, frame_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Обновляем спектрограмму
    spec_mask = (times_spec >= start_time) & (times_spec <= end_time)
    
    if np.any(spec_mask):
        im.set_data(D[:, spec_mask])
        im.set_extent([start_time, end_time, 0, sr/2])
    
    ax1.set_xlim(start_time, end_time)
    # Очищаем предыдущие вертикальные линии
    for line in ax1.lines:
        line.remove()
    # Добавляем новую вертикальную линию
    ax1.axvline(x=current_time, color='white', linewidth=2, alpha=0.7)

    # Устанавливаем метки времени
    tick_step = 0.1
    num_ticks = int(WINDOW_SECONDS / tick_step)
    if start_time > 0 and end_time < frame_idx / fps:
        xticks = np.linspace(start_time, end_time, num_ticks + 1)
    else:
        xticks = np.array([start_time] + list(np.arange(np.ceil(start_time * 10) / 10, end_time, 0.2)) + [end_time])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=90)
    
    # Сохраняем график в изображение
    fig.canvas.draw()
    graph_img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Конвертируем и накладываем на кадр
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    graph_img = cv2.resize(graph_img, (width, height))
    frame = cv2.resize(frame, (width, height))
    combined_frame = np.vstack((frame, graph_img))
    
    # Сохраняем кадр
    frame_path = f"{FRAMES_DIR}/frame_{frame_idx:06d}.png"
    cv2.imwrite(frame_path, combined_frame)
    
    if frame_idx % 100 == 0:
        print(f"Обработано: {frame_idx}/{total_frames}")

# Освобождаем ресурсы
plt.close(fig)
cap.release()

# Создаем видео из кадров
print("Создание видео из кадров...")
os.system(f"ffmpeg -framerate {fps} -i {FRAMES_DIR}/frame_%06d.png -i {AUDIO_PATH} -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -pix_fmt yuv420p {OUTPUT_VIDEO}")

print(f"Готово! Результат сохранен в {OUTPUT_VIDEO}")