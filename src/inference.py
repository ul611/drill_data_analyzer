import os
import torch
import torchaudio
import argparse
import yaml
from types import SimpleNamespace
import numpy as np
import glob
import csv
import datetime
from .train import load_config, MotorClassifier

def load_model_from_checkpoint(checkpoint_path, config):
    """Загрузка модели из чекпоинта"""
    model = MotorClassifier.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    return model

def preprocess_audio(audio_path, config):
    """Предобработка аудиофайла для инференса"""
    # Загрузка аудио
    audio, sr = torchaudio.load(audio_path)
    
    # Ресемплирование, если необходимо
    if sr != config.data.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, config.data.sample_rate)
    
    # Обеспечиваем длину сегмента
    desired_length = int(config.data.segment_length * config.data.sample_rate)
    current_length = audio.shape[1]
    
    if current_length < desired_length:
        # Дополняем аудио нулями, если оно короче требуемой длины
        padding = desired_length - current_length
        audio = torch.nn.functional.pad(audio, (0, padding))
    elif current_length > desired_length:
        # Обрезаем аудио, если оно длиннее требуемой длины
        audio = audio[:, :desired_length]
    
    return audio.unsqueeze(0)  # Добавляем размерность батча (1, channels, time)

def predict(model, audio_tensor, device="cpu"):
    """Получение предсказания модели"""
    model = model.to(device)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        # Используем forward метод напрямую - теперь он есть в MotorClassifier
        logits = model(audio_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].tolist()

def load_class_mapping(config_path):
    """Загрузка маппинга классов из конфигурации или создание дефолтного"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'class_mapping' in config:
            return config['class_mapping']
    except:
        pass
    
    # Дефолтное маппирование - просто индексы
    return {i: f"Class_{i}" for i in range(4)}

def process_audio_file(audio_path, model, config, class_mapping, device):
    """Обработка одного аудиофайла и получение результатов"""
    try:
        audio_tensor = preprocess_audio(audio_path, config)
        predicted_class, confidence, all_probs = predict(model, audio_tensor, device)
        
        result = {
            "path": audio_path,
            "predicted_class": predicted_class,
            "class_name": class_mapping.get(predicted_class, f'Класс_{predicted_class}'),
            "confidence": confidence,
            "probabilities": all_probs
        }
        return result
    except Exception as e:
        print(f"Ошибка при обработке файла {audio_path}: {str(e)}")
        return None

def save_results_to_csv(results, output_file, class_mapping):
    """Сохранение результатов в CSV-файл"""
    if not results:
        print("Нет результатов для сохранения")
        return
    
    # Получаем названия классов для заголовков
    class_headers = [f"prob_{class_mapping.get(i, f'Class_{i}')}" for i in range(len(results[0]['probabilities']))]
    
    # Открываем файл и пишем результаты
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file_path', 'predicted_class', 'class_name', 'confidence'] + class_headers
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=',')
        
        writer.writeheader()
        for result in results:
            row = {
                'file_path': result['path'],
                'predicted_class': result['predicted_class'],
                'class_name': result['class_name'],
                'confidence': f"{result['confidence']:.4f}"
            }
            
            # Добавляем вероятности по всем классам
            for i, prob in enumerate(result['probabilities']):
                row[class_headers[i]] = f"{prob:.4f}"
            
            writer.writerow(row)
    
    print(f"Результаты сохранены в {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Инференс модели для классификации звуков')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Путь к файлу чекпоинта модели (.ckpt)')
    parser.add_argument('--model-type', '-mt', type=str, choices=['cnn14', 'ast'], default='cnn14',
                       help='Тип модели: cnn14 или ast')
    
    # Делаем группу взаимоисключающих аргументов для audio и folder
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio', type=str, help='Путь к аудиофайлу для классификации (.wav)')
    input_group.add_argument('--folder', type=str, help='Путь к папке с аудиофайлами (.wav)')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Путь к файлу конфигурации (опционально)')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения результатов в CSV (по умолчанию: results_YYYYMMDD_HHMMSS.csv)')
    
    args = parser.parse_args()
    
    # Определяем путь к конфигурации модели
    if args.config:
        model_config_path = args.config
    else:
        if args.model_type == 'ast':
            model_config_path = 'drill_data_analyzer/configs/model_ast.yaml'
        else:
            model_config_path = 'drill_data_analyzer/configs/model_cnn14.yaml'
    
    # Загружаем конфигурации
    data_config_path = 'drill_data_analyzer/configs/data.yaml'
    config_paths = [data_config_path, model_config_path]
    
    # Объединяем конфигурации
    config = SimpleNamespace()
    for path in config_paths:
        partial_config = load_config(path)
        for key, value in partial_config.__dict__.items():
            setattr(config, key, value)
    
    # Проверяем существование файла чекпоинта
    if not os.path.exists(args.checkpoint):
        print(f"Ошибка: файл чекпоинта {args.checkpoint} не найден")
        return
    
    # Загружаем модель
    print(f"Загрузка модели из {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Загружаем маппинг классов (если доступен)
    class_mapping = load_class_mapping('drill_data_analyzer/configs/class_mapping.yaml')
    
    # Определяем устройство для инференса
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = []
    
    # Обработка одного файла или папки
    if args.audio:
        # Проверка существования файла
        if not os.path.exists(args.audio):
            print(f"Ошибка: аудиофайл {args.audio} не найден")
            return
        
        # Обрабатываем один файл
        print(f"Загрузка и предобработка аудиофайла {args.audio}...")
        result = process_audio_file(args.audio, model, config, class_mapping, device)
        if result:
            results.append(result)
            
            # Выводим результаты на экран
            print("\n" + "="*50)
            print(f"Файл: {os.path.basename(args.audio)}")
            print(f"Предсказанный класс: {result['class_name']} (индекс: {result['predicted_class']})")
            print(f"Уверенность: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print("\nВероятности по всем классам:")
            for i, prob in enumerate(result['probabilities']):
                print(f"  {class_mapping.get(i, f'Класс_{i}')}: {prob:.4f} ({prob*100:.2f}%)")
            print("="*50)
    
    elif args.folder:
        # Проверка существования папки
        if not os.path.exists(args.folder):
            print(f"Ошибка: папка {args.folder} не найдена")
            return
        
        # Получаем список WAV файлов в папке (включая подпапки)
        wav_files = glob.glob(os.path.join(args.folder, "**/*.wav"), recursive=True)
        
        if not wav_files:
            print(f"В папке {args.folder} не найдено WAV файлов")
            return
        
        print(f"Найдено {len(wav_files)} WAV файлов в папке {args.folder}")
        
        # Обрабатываем каждый файл
        for i, audio_path in enumerate(wav_files):
            print(f"Обработка файла {i+1}/{len(wav_files)}: {os.path.basename(audio_path)}...")
            result = process_audio_file(audio_path, model, config, class_mapping, device)
            if result:
                results.append(result)
    
    # Сохраняем результаты в файл, если есть что сохранять
    if results:
        # Определяем имя выходного файла
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results_{args.model_type}_{timestamp}.csv"
        
        save_results_to_csv(results, output_file, class_mapping)

if __name__ == "__main__":
    main()