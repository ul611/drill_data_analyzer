import librosa

audio, sr = librosa.load("/home/ul/plot_drill/audio_3961.mp3", sr=None)  # sr - частота дискретизации
duration = len(audio) / sr  # Длительность в секундах
total_samples = len(audio)  # Искомое количество измерений

print(duration)
print(total_samples)