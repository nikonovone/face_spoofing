# **Face Antispoofing**

Здесь вы можете ознакомиться с базовой реализацией нейросети с использованием Pytorch Lighting для решения задачи антиспуфинга лица.

Для запуска тренировки:
```
    python train.py --data_dir=PATH_TO_DATASET                    
                    --check_dir=PATH_TO_SAVE_LOGS
                    --num_epochs=NUM_EPOCHS
```

## **Визуализация результатов работы**

<div align='center'>
 <image src="test/output.jpg" height="350" alt="Текст с описанием картинки" caption="Подпись под картинкой"/>
</div>

15 epochs training:

f1_score: **0.885**

precision: **0.920**

recall: **0.852**



## **Зависимости**
Настройка окружения:
   * В первую очередь нужно настроить - [pytorch](https://pytorch.org/) и [pytorch-lighting](https://www.pytorchlightning.ai/) 

   * Далее установить все зависимости из файла requirements.txt, с помощью команды
     ```shell
     pip install -r requirements.txt
     ```

## **Структура датасета**

Структура входных данных для сети
```
├── Dataset_folder/
│   ├──  train/
│           ├── 0001.jpg
|           ├── 0002.jpg
|           ...
│   ├──  test/ 
│           ├── 0001.jpg
|           ├── 0002.jpg
|           ... 
│   ├──  test.csv
│   ├──  train.csv
```

Структура .csv файлов
| image               | spoof_type   | spoof_label    | box                                              |
|---------------------|--------------|----------------|--------------------------------------------------|
| Путь до изображения | Тип спуфинга | Бинарная метка | Бокс лица в формате [x_min, y_min, x_max, y_max] |

## **Описание нейронной сети**
В работе используется предобученная [ResNet18](https://rwightman.github.io/pytorch-image-models/models/resnet/).

С функцией-потерь [Focal-Loss](https://paperswithcode.com/method/focal-loss).

Данные аугментировались случайной гаммой, горизонтальным отражением,
небольшим случайным поворотом и случайными яркостью, контрастностью.

На вход сеть получает обрезанное по координатам box исходное ихображение,
и на выходе выдает метку 0 или 1.

В конце выводятся классические метрики дли бинарной классификации:

* **f-score**
* **recall**
* **precision**

Акцент в данной задаче стоить сделать на **recall** для минимизации ложно-отрицательных срабатываний классификатора (к примеру можно использовать **f-beta-score**).

### Данные использующиеся в baseline.
Отмечу небольшую несбаланнсированность обучающего датасета, а также крайнюю несбалансированность тестового датасета.

## **Описание системы логирования**
В процессе тренировки, происходит логирование с помощью [Tensorboard](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html) через pytorch-lighting.
Все логи сохраняются в папкe **checkpoints**.

Структура **checkpoints**.
```
├── checkpoints/
│   ├──  version_0/                   # Папка с вашим экспериментом
│           ├── checkpoints/
|                   ├── name.ckpt     # Файл модели
|           ├── tb_logs/
|                   ├── events.out.tfevents.1671447233.namePC
|                   ...
|                   ├── hparams.yaml  # Файл гиперпараметров
│   ├──  version_1
│           ...
```
