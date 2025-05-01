import os
import cv2
import numpy as np
import pandas as pd
from scipy.io import wavfile
from pathlib import Path
from tqdm.notebook import tqdm

def get_frame_timestamps_robust(video_path, start_frame, end_frame):
    """
    Get exact timestamps for frames using frame index and FPS.
    This matches the approach used in video_and_spec_with_frames.py.
    
    Args:
        video_path (str): Path to the video file
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        
    Returns:
        tuple: (start_time, end_time) in seconds, or (None, None) if failed
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print(f"Invalid frame range: {start_frame}-{end_frame} for video with {total_frames} frames")
        cap.release()
        return None, None
    
    # Calculate timestamps using frame index and FPS
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    # Add a small safety margin to ensure we don't miss any audio
    safety_margin = 0.1  # 100ms margin
    start_time = max(0, start_time - safety_margin)
    
    print(f"Frame {start_frame} -> {start_time:.3f}s (FPS: {fps:.2f})")
    print(f"Frame {end_frame} -> {end_time:.3f}s (FPS: {fps:.2f})")
    
    cap.release()
    return start_time, end_time

def extract_audio_segment(video_number, start_frame, end_frame, motor_state, motor_type, 
                         video_dir, audio_dir, output_dir):
    """
    Extract audio segment based on video frame range using robust timestamp method.
    
    Args:
        video_number (int): Video number
        start_frame (int): Starting frame number
        end_frame (int): Ending frame number
        motor_state (str): Motor state
        motor_type (str): Motor type
        video_dir (str): Directory containing video files
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory to save extracted audio segments
        
    Returns:
        str: Path to the saved audio segment, or None if failed
    """
    # Construct file paths
    video_path = os.path.join(video_dir, f'IMG_{video_number}.MOV')
    audio_path = os.path.join(audio_dir, f'audio_{video_number}_stereo.wav')
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None
    
    # Get exact timestamps for the frames using the robust method
    start_time, end_time = get_frame_timestamps_robust(video_path, start_frame, end_frame)
    
    if start_time is None or end_time is None:
        return None
    
    # Read audio file
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert time to samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure we don't go out of bounds
    if start_sample >= len(audio_data):
        print(f"Start sample {start_sample} is beyond audio length {len(audio_data)}")
        return None
        
    # Adjust end_sample if it exceeds audio length
    if end_sample > len(audio_data):
        print(f"Adjusting end sample from {end_sample} to {len(audio_data)}")
        end_sample = len(audio_data)
        
    if start_sample >= end_sample:
        print(f"Invalid sample range: start_sample ({start_sample}) >= end_sample ({end_sample})")
        return None
    
    # Extract audio segment
    audio_segment = audio_data[start_sample:end_sample]
    
    # Create output directory if it doesn't exist
    output_subdir = os.path.join(output_dir, motor_type)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Create filename with all necessary information
    filename = f"{video_number}_{start_frame}_{end_frame}_{motor_state}.wav"
    output_path = os.path.join(output_subdir, filename)
    
    # Save audio segment
    wavfile.write(output_path, sample_rate, audio_segment)
    
    return output_path

def process_segments(segments_df, video_dir, audio_dir, output_dir):
    """
    Process all segments in the dataframe and extract audio segments.
    
    Args:
        segments_df (pd.DataFrame): DataFrame containing segment information
        video_dir (str): Directory containing video files
        audio_dir (str): Directory containing audio files
        output_dir (str): Directory to save extracted audio segments
        
    Returns:
        pd.DataFrame: DataFrame containing information about processed segments
    """
    processed_files = []
    
    for _, row in tqdm(segments_df.iterrows(), total=len(segments_df), desc="Processing segments"):
        video_number = row['video_number']
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        motor_state = row['motor_state']
        motor_type = row['motor_type']
        
        output_path = extract_audio_segment(
            video_number, 
            start_frame, 
            end_frame, 
            motor_state, 
            motor_type,
            video_dir,
            audio_dir,
            output_dir
        )
        
        if output_path:
            processed_files.append({
                'video_number': video_number,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'motor_state': motor_state,
                'motor_type': motor_type,
                'audio_file': output_path
            })
    
    return pd.DataFrame(processed_files) 

import os
import shutil

# Base directories
base_dir = '/home/ul/plot_drill/drill_data_analyzer/'
source_dir = os.path.join(base_dir, 'artefacts/plots/plot_for_video/')
target_base_dir = source_dir

# Define motor groups and their corresponding video numbers
motor_groups = {
    '4т_двигатель': [4012, 4013, 4039, 4040, 4125, 4126],
    '2т_двигатель': [4116, 4117, 4118, 4119, 4121, 4123, 4124, 4137],
    'щеточный_электрический_двигатель': [4128, 4129, 4130, 4132, 4136],
    'BL_мотор': [4133, 4134]
}

import librosa
import pandas as pd

# Function to analyze audio duration
def get_audio_duration(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr  # Duration in seconds
        return duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Base directory for audio files
audio_base_dir = os.path.join(base_dir, 'data/audios/')

# Dictionary to store results
results = {}

# Process each motor group
for group_name, video_numbers in motor_groups.items():
    print(f"\n{group_name}:")
    group_total_duration = 0
    group_results = []
    
    for video_num in video_numbers:
        audio_path = os.path.join(audio_base_dir, f'audio_{video_num}_stereo.wav')
        
        if os.path.exists(audio_path):
            duration = get_audio_duration(audio_path)
            if duration is not None:
                group_total_duration += duration
                print(f"  Видео {video_num} - {duration:.2f} секунд")
                group_results.append({"Видео": video_num, "Длительность (сек)": round(duration, 2)})
        else:
            print(f"  Видео {video_num} - файл не найден")
            group_results.append({"Видео": video_num, "Длительность (сек)": "Файл не найден"})
    
    print(f"  Итого для группы: {group_total_duration:.2f} секунд")
    results[group_name] = {"data": group_results, "total": round(group_total_duration, 2)}

for group_name, group_data in results.items():
    print(f"\n{group_name}:")
    df = pd.DataFrame(group_data["data"])
    print(f"Итого длительность: {group_data['total']} секунд")


import pandas as pd
# Parse video segments with motor states
video_segments = []

# Process each line of data
for line in """4012
0-12866 холостые 
4013
0-12458 холостые 
4039
0-7641 холостые
4040
0-336 холостые 
2047-3957 максимальные 
4147-4852 холостые 
4936-5750 набор 
5810-7516 максимальные 
7516-8031 сброс 
4125
0-2460 холостые 
2467-2572 набор 
2572-4900 меньше среднего 
4960-5196 сброс 
5196-7182 холостые 
7206-7481 набор 
7481-9692 выше среднего 
9700-9997 сброс 
10000-12044 холостые 
12070-12580 набор 
12600-17089 максимальные
4126
0-5166 холостые
4121
0-261 разгон 
261-2505 максимальные 
2515-2685 сброс 
2750-3151 холостые 
3172-3216 разгон 
3216-3388 затухание 
3388-4900 холостые 
4953-5047 разгон 
5047-7283 максимальные 
7290-7487 сброс 
7500-7743 холостые
4123
0-2476 холостые 
2479-2527 набор 
2527-2630 сброс 
2630-2829 выше среднего 
2997-4629 средние 
4639-4934 сброс 
4124
578-724 средние 
724-1017 сброс 
1017-2486 холостые 
2520-2714 набор 
2714-7287 высокие 
7295-7604 сброс 
8003-8454 холостые 
8874-9701 холостые 
9728-9956 набор 
9956-12267 максимальные 
12267-12601 сброс 
12601-15937 холостые
4116
0-2640 холостые 
2640-4904 средние 
5242-10043 холостые 
10043-12099 максимальные 
12099-12848 сброс оборотов 
12848-14535 холостые 
4117
0-2522 холостые 
2522-3168 выше среднего 
3247-4963 средние 
5056-7304 холостые 
7304-9719 максимальные 
10097-12121 холостые 
12253-14533 максимальные 
14540-14770 сброс на холостые
4118
0-2546 холостые 
2544-2618 средние 
2700-4929 средние
4944-5490 холостые 
4119
0-1996 холостые 
1996-2068 средние 
2100-2481 холостые 
2503-2596 выше среднего 
2649-2858 выше среднего 
2880-4884 ниже среднего 
4884-5299 холостые
4132
122-247 разгон 
247-2436 высокие 
2440-2824 снижение 
2824-5028 средние 
5047-5209 разгон 
5209-7340 высокие 
7566-7705 разгон 
7710-9656 средние
9685-9791 разгон 
9791-12091 высокие
4134
0-9690 высокие
4136
220-348 разгон 
436-9732 высокие (стандартные) 
4133
108-168 разгон 
168-2508 высокие 
2520-2781 снижение 
2781-4877 средние
4879-4938 разгон 
4939-7359 высокие 
7503-7859 ниже среднего
7892-7919 снижение 
7919-9695 средние 
9721-9755 разгон 
9767-12104 высокие
12105-12353 снижение
12354-12517 низкие
12518-12778 набор
12779-12978 высокие
12979-13326 снижение
13327-13400 низкие
13401-13620 набор
13621-13738 высокие
13739-14100 снижение
14101-14292 низкие
14293-14438 набор
14439-14560 высокие
14561-14613 сброс
4137
0-2739 холостые 
2748-3043 набор 
3060-4903 максимальные 
4903-5694 сброс 
5694-7311 холостые 
7311-7919 набор 
7919-9735 средние 
9745-10134 сброс 
10137-12135 холостые 
12143-12296 набор
12296-13131 максимальные 
4128
261-710 набор 
724-14573 рабочие
4129
396-537 набор 
537-2611 максимальные 
3266-4861 средние 
4867-4948 набор 
4987-7302 максимальные 
7320-8412 сброс 
8447-9366 средние 
9999-10093 набор 
10093-12303 максимальные 
12347-13520 сброс 
13520-14693 средние
4130
396-936 набор
936-14475 максимальные (рабочие)
14475-15303 сброс""".strip().split('\n'):
    line = line.strip()
    if not line:
        continue
    
    # Check if this is a video number line
    if line.isdigit():
        current_video = int(line)
    else:
        # This is a segment line
        parts = line.split(' ', 1)
        if len(parts) == 2:
            frame_range, motor_state = parts
            
            # Parse frame range
            if '-' in frame_range:
                start_frame, end_frame = map(int, frame_range.split('-'))
                
                # Add to segments list
                video_segments.append({
                    'video_number': current_video,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'motor_state': motor_state
                })

# Convert to DataFrame for better visualization
segments_df = pd.DataFrame(video_segments)

# Optional: Group by video number and motor state for analysis
grouped_stats = segments_df.groupby(['motor_state']).agg({
    'start_frame': 'count',
   }).rename(columns={'start_frame': 'segment_count'})


# stable_segments_df = segments_df.query('motor_state not in ["затухание", "набор", "разгон", "сброс", "сброс оборотов", "сброс на холостые", "снижение"]').copy()
stable_segments_df = segments_df.query('motor_state in ["затухание", "набор", "разгон", "сброс", "сброс оборотов", "сброс на холостые", "снижение"]').copy()

# Create a mapping of video numbers to motor types
motor_type_mapping = {}

for motor_type, motor_data in results.items():
    for video_info in motor_data['data']:
        video_number = video_info['Видео']
        motor_type_mapping[video_number] = motor_type

# Add motor type column to the stable segments dataframe
stable_segments_df['motor_type'] = stable_segments_df['video_number'].map(motor_type_mapping)


# Define base paths
VIDEO_DIR = '/home/ul/plot_drill/drill_data_analyzer/data/videos/'
AUDIO_DIR = '/home/ul/plot_drill/drill_data_analyzer/data/audios/'
# OUTPUT_DIR = '/home/ul/plot_drill/drill_data_analyzer/data/raw_data_to_train/'
OUTPUT_DIR = '/home/ul/plot_drill/drill_data_analyzer/data/raw_data_to_infer/'

# Process segments using the new robust method
processed_df = process_segments(
    stable_segments_df,
    VIDEO_DIR,
    AUDIO_DIR,
    OUTPUT_DIR
)

# Display results
print(processed_df)

# Print summary
print(f"Extracted {len(processed_df)} audio segments")
print(f"Segments by motor type:")
print(processed_df.groupby('motor_type').size())
print(f"Segments by motor state:")
print(processed_df.groupby('motor_state').size())