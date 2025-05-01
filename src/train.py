import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
from .models.cnn14 import CNN14
from .models.ast import AudioModel as ASTModel
from .data_module import MotorDataModule
import yaml
from types import SimpleNamespace
import os
import copy
import argparse
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from easydict import EasyDict

def load_model(config):
    """Загрузка модели на основе конфигурации"""
    if config.model.name == "CNN14":
        return CNN14(config)
    elif config.model.name == "AST":
        return ASTModel(config)
    else:
        raise ValueError(f"Неизвестная модель: {config.model.name}")

class MotorClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = load_model(config)  # CNN14, AST и т.д.
        self.criterion = torch.nn.CrossEntropyLoss()
        # Сохраняем гиперпараметры, игнорируя сложный объект config
        self.save_hyperparameters(ignore=['config'])

    def forward(self, x):
        """Прямой проход через модель"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        # Преобразуем learning_rate в float, если он строка
        learning_rate = self.config.model.learning_rate
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

def dict_to_namespace(d):
    """Рекурсивно преобразует dict в SimpleNamespace"""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            # Преобразуем строки, которые выглядят как числа, в числа
            if isinstance(value, str):
                try:
                    # Пробуем преобразовать в int
                    int_value = int(value)
                    setattr(namespace, key, int_value)
                except ValueError:
                    try:
                        # Пробуем преобразовать в float
                        float_value = float(value)
                        setattr(namespace, key, float_value)
                    except ValueError:
                        # Если не удалось, оставляем как строку
                        setattr(namespace, key, value)
            else:
                setattr(namespace, key, value)
    return namespace

def load_config(config_path):
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Рекурсивно преобразуем словарь в объект с доступом через точку
    config = dict_to_namespace(config_dict)
    
    # Добавляем метод copy
    def namespace_copy(self):
        return copy.deepcopy(self)
    
    # Добавляем метод copy ко всем вложенным объектам
    for key, value in config.__dict__.items():
        if isinstance(value, SimpleNamespace):
            setattr(value, 'copy', namespace_copy.__get__(value))
    
    setattr(config, 'copy', namespace_copy.__get__(config))
    
    return config

def train(config_paths):
    """Объединение конфигураций и запуск обучения"""
    # Загружаем и объединяем конфигурации
    config = SimpleNamespace()
    for path in config_paths:
        partial_config = load_config(path)
        for key, value in partial_config.__dict__.items():
            setattr(config, key, value)
    
    # Создаем директорию для сохранения чекпоинтов, если она не существует
    checkpoint_dir = os.path.join("drill_data_analyzer", "checkpoints", config.model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Настройка callbacks для сохранения моделей
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    wandb.init(project="motor-sound-classification", config=vars(config))
    datamodule = MotorDataModule(config)
    model = MotorClassifier(config)
    
    # Используем современный API PyTorch Lightning
    if torch.cuda.is_available():
        # Для GPU
        trainer = pl.Trainer(
            logger=WandbLogger(),
            max_epochs=config.epochs if hasattr(config, 'epochs') else 10,
            accelerator='gpu',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor]
        )
    else:
        # Для CPU
        trainer = pl.Trainer(
            logger=WandbLogger(),
            max_epochs=config.epochs if hasattr(config, 'epochs') else 10,
            accelerator='cpu',
            devices=1,  # Для CPU нужно указать конкретное число (не None)
            callbacks=[checkpoint_callback, lr_monitor]
        )
    
    trainer.fit(model, datamodule)
    
    # Сохраняем лучшую модель дополнительно с удобным именем
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Лучшая модель сохранена по пути: {best_model_path}")
        
        # Создаем понятное имя файла для лучшей модели
        model_name = f"{config.model.name}_best.ckpt"
        best_model_final_path = os.path.join(checkpoint_dir, model_name)
        
        # Копируем лучшую модель с более понятным именем
        import shutil
        shutil.copy(best_model_path, best_model_final_path)
        print(f"Копия лучшей модели сохранена как: {best_model_final_path}")
    
    return model, checkpoint_callback.best_model_path

def main(args):
    with open(args.config_path) as f:
        config = EasyDict(yaml.safe_load(f))
    
    # Создаем директорию для сохранения моделей, если она не существует
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    # Загружаем модель на основе конфигурации
    if config.model.name == "AST":
        model_class = ASTModel
    else:
        model_class = CNN14
    
    # Создаем модель
    model = model_class(config)
    
    # Создаем Lightning модуль
    lightning_model = MotorClassifier(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели для анализа звуков')
    parser.add_argument('--model-type', '-mt', type=str, choices=['cnn14', 'ast'], default='cnn14',
                        help='Тип модели: cnn14 или ast')
    args = parser.parse_args()
    
    config_paths = [
        "drill_data_analyzer/configs/data.yaml",
    ]
    
    # Выбор конфигурации модели на основе аргумента
    if args.model_type == 'ast':
        config_paths.append("drill_data_analyzer/configs/model_ast.yaml")
    else:
        config_paths.append("drill_data_analyzer/configs/model_cnn14.yaml")
    
    train(config_paths)
