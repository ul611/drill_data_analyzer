import os
import torchaudio
from torch.utils.data import Dataset
import torch
import numpy as np

class MotorSoundDataset(Dataset):
    def __init__(self, config):
        self.config = config  # Сохраняем конфигурацию для возможного использования в подклассах
        self.sample_rate = config.data.sample_rate
        self.segment_length = config.data.segment_length
        self.augment = config.data.augment
        
        # Выводим информацию о пути к данным
        data_path = config.data.paths.train
        print(f"Загрузка данных из: {data_path}, абсолютный путь: {os.path.abspath(data_path)}")
        print(f"Директория существует: {os.path.exists(data_path)}")
        
        self.filepaths, self.labels, self.class_to_idx = self._load_data(data_path)
        print(f"Загружено файлов: {len(self.filepaths)}")
        if len(self.filepaths) > 0:
            print(f"Пример пути к файлу: {self.filepaths[0]}")
            print(f"Классы: {self.class_to_idx}")
        else:
            print("ВНИМАНИЕ: Не найдено ни одного WAV файла!")

    def _load_data(self, data_dir):
        # Загрузка путей к файлам и меток
        filepaths = []
        labels = []
        
        # Проверяем, существует ли директория
        if not os.path.exists(data_dir):
            print(f"ОШИБКА: Директория {data_dir} не существует!")
            return filepaths, labels, {}
        
        # Получаем уникальные метки (имена папок)
        unique_labels = []
        for root, dirs, files in os.walk(data_dir):
            print(f"Сканирование {root}, найдено {len(dirs)} поддиректорий и {len(files)} файлов")
            for dir_name in dirs:
                unique_labels.append(dir_name)
        
        print(f"Найдены поддиректории (возможные классы): {unique_labels}")
        
        # Создаем словарь для преобразования меток в индексы
        class_to_idx = {label: i for i, label in enumerate(sorted(unique_labels))}

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    filepaths.append(os.path.join(root, file))
                    label_name = os.path.basename(root)
                    labels.append(class_to_idx.get(label_name, 0))  # Преобразуем метку в индекс

        return filepaths, labels, class_to_idx

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.filepaths[idx])
        print(f"Оригинальная форма аудио после загрузки: {audio.shape}, sr={sr}")
        
        audio = self._preprocess(audio, sr)
        print(f"Форма аудио после предобработки: {audio.shape}")
        
        if self.augment:
            audio = self._augment(audio)
            print(f"Форма аудио после аугментации: {audio.shape}")
        
        # Проверяем, что форма правильная: [channels, time]
        if len(audio.shape) != 2:
            raise ValueError(f"Неверная форма аудио: {audio.shape}")
        
        return audio, self.labels[idx]
    
    def _preprocess(self, audio, sr):
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        
        # Обеспечиваем длину сегмента
        desired_length = int(self.segment_length * self.sample_rate)
        current_length = audio.shape[1]
        
        if current_length < desired_length:
            # Дополняем аудио нулями, если оно короче требуемой длины
            padding = desired_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif current_length > desired_length:
            # Обрезаем аудио, если оно длиннее требуемой длины
            audio = audio[:, :desired_length]
            
        return audio

    def _augment(self, audio):
        # Добавление шума, pitch-shift и т.д.
        if np.random.rand() > 0.5:
            audio += torch.randn_like(audio) * 0.005
        return audio
