#!/usr/bin/env python
import os
import sys
import argparse

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drill_data_analyzer.src.inference import main as inference_main

if __name__ == "__main__":
    inference_main() 