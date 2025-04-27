import argparse
import subprocess

# Local imports
from configs import config

def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract stereo audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
    """
    subprocess.run([
        'ffmpeg',
        '-i', video_path,        # Input video file
        '-vn',                   # Disable video recording
        '-acodec', 'pcm_s16le',  # Use PCM 16-bit audio codec
        '-ar', '44100',          # Set audio sample rate to 44.1kHz
        '-ac', '2',              # Set 2 audio channels (stereo)
        audio_path,              # Output audio file
        '-y'                     # Overwrite output file if exists
    ], check=True)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract audio from video file')
    parser.add_argument('--video-number', '-v', 
                       type=int, 
                       required=True,
                       help='Video number to process')
    args = parser.parse_args()

    # Construct input/output paths
    video_path = os.path.join(config.basedir, 'data', 'videos', f'IMG_{args.video_number}.MOV')
    audio_path = os.path.join(config.basedir, 'data', 'audios', f'audio_{args.video_number}_stereo.wav')

    # Extract audio from video
    extract_audio(video_path, audio_path)


if __name__ == '__main__':
    main()
