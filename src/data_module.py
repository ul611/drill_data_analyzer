import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .data_loader import MotorSoundDataset
import copy
import os
import torch

def custom_collate_fn(batch):
    """
    Кастомная функция для формирования батчей,
    которая обеспечивает правильную форму тензоров.
    """
    # Извлекаем аудио и метки из батча
    audios, labels = zip(*batch)
    
    # Проверяем форму аудио
    print(f"Аудио в батче: {len(audios)}, форма первого элемента: {audios[0].shape}")
    
    # Объединяем в тензоры
    audios = torch.stack(audios)
    labels = torch.tensor(labels)
    
    # print(f"Форма батча аудио: {audios.shape}, метки: {labels.shape}")
    
    return audios, labels

class MotorDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.data.batch_size
        
        # Выводим информацию о конфигурации
        print(f"Конфигурация DataModule:")
        print(f"  Путь к тренировочным данным: {config.data.paths.train}")
        print(f"  Путь к валидационным данным: {config.data.paths.val}")
        print(f"  Частота дискретизации: {config.data.sample_rate} Гц")
        print(f"  Длина сегмента: {config.data.segment_length} сек")
        print(f"  Размер батча: {config.data.batch_size}")
        print(f"  Аугментация: {'включена' if config.data.augment else 'выключена'}")

    def setup(self, stage=None):
        # Датасет для обучения
        print("Инициализация тренировочного датасета...")
        self.train_dataset = MotorSoundDataset(self.config)
        
        # Создаем новый экземпляр датасета для валидации
        # с отключенной аугментацией и путем к валидационным данным
        print("Инициализация валидационного датасета...")
        val_config = copy.deepcopy(self.config)
        val_config.data.augment = False
        
        # Создаем отдельный датасет для валидации
        # Напрямую меняем путь в config, это безопаснее
        val_config.data.paths.train = val_config.data.paths.val
        self.val_dataset = MotorSoundDataset(val_config)

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError("Тренировочный датасет пуст! Проверьте путь к данным.")
        
        print(f"Создание DataLoader для тренировочного датасета с {len(self.train_dataset)} образцами")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        if len(self.val_dataset) == 0:
            print("ПРЕДУПРЕЖДЕНИЕ: Валидационный датасет пуст! Возвращаем None.")
            return None
        
        print(f"Создание DataLoader для валидационного датасета с {len(self.val_dataset)} образцами")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
        ) 