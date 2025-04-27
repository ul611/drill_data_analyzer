import argparse
import numpy as np
import cv2
import librosa
import matplotlib.pyplot as plt
import librosa.display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import gc
from configs.config import basedir, log_to_video, video_audio_ms_deltas, window_seconds, graph_height


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process video and generate spectrograms')
    parser.add_argument('--video-number', '-v', type=int, required=True, help='Video number to process')
    return parser.parse_args()


def setup_paths(video_number):
    """Setup paths for input and output files."""
    
    video_path = os.path.join(basedir, f"data/videos/IMG_{video_number}.MOV")
    audio_path = os.path.join(basedir, f"data/audios/audio_{video_number}_stereo.wav")
    output_video = os.path.join(basedir, f"artefacts/videos_with_spec/video_with_frames_and_spec_{video_number}.mp4")
    frames_dir = os.path.join(basedir, f"artefacts/output_frames/output_frames_{video_number}")
    
    return video_path, audio_path, output_video, frames_dir, video_number


def clean_plt():
    """Free matplotlib resources."""
    plt.close('all')
    gc.collect()


def initialize_video(video_path, ratio=0.7):
    """Initialize video capture and get video properties."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Check if video is vertical (height > width)
    if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > cap.get(cv2.CAP_PROP_FRAME_WIDTH):
        # For vertical video, swap width and height and rotate later
        height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio)
        width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
        is_vertical = True
    else:
        # For horizontal video, keep original dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * ratio)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
        is_vertical = False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, fps, width, height, total_frames, is_vertical


def load_audio(audio_path):
    """Load audio file and compute spectrogram."""
    print("Загрузка аудио...")
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, window='hamming')), ref=np.max)
    times_spec = librosa.times_like(D, sr=sr)
    
    return y, sr, D, times_spec


def create_figure():
    """Create matplotlib figure for spectrogram visualization."""
    print("Создание фигуры...")
    plt.switch_backend('Agg')
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10), 
                          sharex=True, gridspec_kw={'hspace': 0.05}, dpi=100)

    # Setup axes
    ax1.set_ylabel('Частота (Гц)')
    ax1.set_ylim(0, 8000)
    divider = make_axes_locatable(ax1)
    
    return fig, ax1


def process_frame(frame, frame_idx, fps, width, height, current_time, window_seconds, total_frames, 
                  ax1, im, D, times_spec, sr, is_vertical):
    """Process a single video frame and update spectrogram."""
    # Rotate frame if video is vertical
    if is_vertical:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Calculate window boundaries to keep current time in center
    half_window = window_seconds / 2
    start_time = max(0, min(current_time - half_window, total_frames / fps - window_seconds))
    end_time = min(max(window_seconds, current_time + half_window), total_frames / fps)
    
    # Add frame number to left bottom corner
    frame_text = f"Frame: {frame_idx}"
    (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, height - text_height - 30), (10 + text_width + 20, height - 10), (255, 255, 255), -1)
    cv2.putText(frame, frame_text, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Update spectrogram
    spec_mask = (times_spec >= start_time) & (times_spec <= end_time)
    
    if np.any(spec_mask):
        im.set_data(D[:, spec_mask])
        im.set_extent([start_time, end_time, 0, sr/2])
    
    ax1.set_xlim(start_time, end_time)
    
    # Clear previous vertical lines
    for line in ax1.lines:
        line.remove()
    
    # Add new vertical line at current time
    ax1.axvline(x=current_time, color='white', linewidth=2, alpha=0.7)

    # Set time ticks
    tick_step = 0.1
    num_ticks = int(window_seconds / tick_step)
    if start_time > 0 and end_time < frame_idx / fps:
        xticks = np.linspace(start_time, end_time, num_ticks + 1)
    else:
        xticks = np.array([start_time] + list(np.arange(np.ceil(start_time * 10) / 10, end_time, 0.2)) + [end_time])
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=90)
    
    return frame, start_time, end_time


def create_combined_frame(frame, fig, width, height):
    """Create combined frame with video and spectrogram."""
    # Save plot to image
    fig.canvas.draw()
    graph_img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Convert and overlay on frame
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
    graph_img = cv2.resize(graph_img, (width, height))
    frame = cv2.resize(frame, (width, height))
    combined_frame = np.vstack((frame, graph_img))
    
    return combined_frame


def process_video(cap, fps, width, height, total_frames, is_vertical, frames_dir, D, times_spec, sr, window_seconds):
    """Process video, creating frames with spectrograms."""
    print("Обработка кадров...")
    
    # Create figure and axes once
    fig, ax1 = create_figure()
    
    # Create empty imshow for later updates
    im = ax1.imshow(np.zeros((D.shape[0], 1)), aspect='auto', origin='lower',
                  cmap='magma', extent=[0, 1, 0, sr/2],
                  vmin=-80, vmax=0)
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        # Process current frame
        frame, start_time, end_time = process_frame(
            frame, frame_idx, fps, width, height, current_time, window_seconds, 
            total_frames, ax1, im, D, times_spec, sr, is_vertical
        )
        
        # Create and save combined frame
        combined_frame = create_combined_frame(frame, fig, width, height)
        frame_path = f"{frames_dir}/frame_{frame_idx:06d}.png"
        cv2.imwrite(frame_path, combined_frame)
        
        if frame_idx % 100 == 0:
            print(f"Обработано: {frame_idx}/{total_frames}")
    
    # Free resources
    plt.close(fig)
    cap.release()


def create_output_video(frames_dir, fps, audio_path, output_video):
    """Create output video from frames with audio."""
    print("Создание видео из кадров...")
    os.system(f"ffmpeg -framerate {fps} -i {frames_dir}/frame_%06d.png -i {audio_path} -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -pix_fmt yuv420p {output_video}")


def main():
    """Main function to process video and generate spectrograms."""
    # Parse arguments
    args = parse_arguments()
    video_number = args.video_number
    
    # Setup paths
    video_path, audio_path, output_video, frames_dir, video_number = setup_paths(video_number)

    print("window_seconds:", window_seconds, "graph_height:", graph_height)
    
    # Ensure frames directory exists
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize video
    cap, fps, width, height, total_frames, is_vertical = initialize_video(video_path)
    
    # Load audio and compute spectrogram
    y, sr, D, times_spec = load_audio(audio_path)
    
    # Process video
    process_video(cap, fps, width, height, total_frames, is_vertical, frames_dir, D, times_spec, sr, window_seconds)
    
    # Create output video
    create_output_video(frames_dir, fps, audio_path, output_video)
    
    print(f"Готово! Результат сохранен в {output_video}")


if __name__ == "__main__":
    main()