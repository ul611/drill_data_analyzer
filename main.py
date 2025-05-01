#!/usr/bin/env python
import os
import sys
import argparse

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drill_data_analyzer.src.train import train

def main():
    parser = argparse.ArgumentParser(description='Обучение модели для распознавания звуков двигателя')
    parser.add_argument('--data-config', type=str, default='drill_data_analyzer/configs/data.yaml',
                        help='Путь к конфигурации данных')
    parser.add_argument('--model-config', '-mc', type=str, default='drill_data_analyzer/configs/model_cnn14.yaml',
                        help='Путь к конфигурации модели')
    parser.add_argument('--model-type', '-mt', type=str, choices=['cnn14', 'ast'], default='cnn14',
                       help='Тип модели: cnn14 или ast')
    
    args = parser.parse_args()
    
    # Используем AST модель, если указано
    if args.model_type == 'ast':
        args.model_config = 'drill_data_analyzer/configs/model_ast.yaml'
    
    config_paths = [args.data_config, args.model_config]
    model, best_model_path = train(config_paths)
    
    print("\n" + "="*50)
    print(f"Обучение завершено!")
    if best_model_path:
        print(f"Лучшая модель: {best_model_path}")
        print(f"Тип модели: {args.model_type}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 