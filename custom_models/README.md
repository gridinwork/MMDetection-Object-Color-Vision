# Как добавить свою модель

Первая версия студии работает на моделях COCO из MMDetection: человек, чашка, стул, ноутбук, бутылка и остальные 80 классов. Свои классы, например `3D_PRINTER`, `FILAMENT_SPOOL`, `PCB`, `ESP32`, `ARDUINO`, `CALIPER`, `SCREWDRIVER`, `PRINTED_PART`, `PACKAGE`, `FLOWER`, подключаются отдельным конфигом и чекпоинтом. Обучать модель из окна программы не нужно.

## Куда положить данные

```
datasets/custom/images/train
datasets/custom/images/val
datasets/custom/annotations/instances_train.json
datasets/custom/annotations/instances_val.json
```

Аннотации — в формате COCO. Папка `training/` предназначена для конфигов и журналов обучения.

## Обучение

Студия ставит пакет `mmdet`, но скрипты обучения живут в репозитории MMDetection. Один раз скачайте его рядом с проектом и запускайте тем же виртуальным окружением:

```bat
git clone https://github.com/open-mmlab/mmdetection.git
.venv\Scripts\python.exe mmdetection\tools\train.py training\my_rtmdet.py
```

За основу конфига возьмите `rtmdet_tiny_8xb32-300e_coco.py` из пакета MMDetection: замените классы, пути к датасету и число классов в голове модели. После обучения сохраните чекпоинт, например `custom_models/printer.pth`, и сам файл конфига `custom_models/printer.py`.

## Регистрация в студии

Добавьте объект в `custom_models/registry.json`:

```json
[
  {
    "model_id": "workshop_printer",
    "display_name": "Workshop printer",
    "task": "detection",
    "config_path": "custom_models/printer.py",
    "checkpoint_path": "custom_models/printer.pth",
    "input_size": 640,
    "supports_masks": false,
    "description": "Local detector for printer parts"
  }
]
```

Для модели с масками укажите `"task": "instance_segmentation"` и `"supports_masks": true`.

Перезапустите студию. Модель появится в Model Manager и в списке моделей. Кнопка DOWNLOAD для неё не нужна: чекпоинт уже лежит в проекте. После этого распознавание снова работает без интернета.

`config_path` — это путь, который понимает `DetInferencer`. Для встроенных моделей студия передаёт имя из metafile MMDetection, для своих — путь к вашему `.py`.
