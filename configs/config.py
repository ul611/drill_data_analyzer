import os

# Используем относительный путь или определяем через переменную окружения
basedir = os.getenv('DRILL_DATA_DIR', os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Mapping dictionaries
log_to_video = {
    46: 3974, 47: 3975, 48: 3976,
    1: 4010, 2: 4011, 3: 4012, 4: 4013,
    5: 4014, 6: 4015, 7: 4017
}

prominences = {
    1: 0.25, 2: 0.23, 3: 0.25, 4: 0.3,
    5: 0.26, 6: 0.27, 7: 0.25
}

video_audio_ms_deltas = {
    1: 0.63, 2: -0.04, 3: -0.02, 4: 0.25,
    5: 0.47, 6: 0.72, 7: 0.63
}


# video_and_spec_with_frames
window_seconds = 10.0  # Window width in seconds
graph_height = 600 