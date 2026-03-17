# faces

Приложение для Orange Pi 5 Max (RK3588): камера смотрит на вход, узнаёт людей по лицу и приветствует их по имени через espeak-ng. Детекция и распознавание — на NPU (модели SCRFD + ArcFace в формате RKNN).

## Что внутри

- `main.py` — основное приложение (один файл, вся логика)
- `known_faces/` — база лиц: одна папка на человека, внутри — фото (`.jpg` / `.png`)
- `models/` — RKNN-модели (скачиваются скриптом `download_models.sh`)
- `data/` — рантайм-данные (создаётся автоматически)
  - `faces.db` — SQLite: лог событий + снапшоты (BLOB), авто-прунинг до 15 записей
- `install.sh` — установка системных зависимостей и Python-окружения на Orange Pi
- `download_models.sh` — загрузка моделей SCRFD и ArcFace из rknn_model_zoo
- `requirements.txt` — Python-пакеты (opencv, numpy, Pillow и т.д.)

## Быстрый старт

```bash
# 1. Установить зависимости (системные пакеты + venv + pip)
chmod +x install.sh && ./install.sh

# 2. RKNN runtime не в PyPI — скачать .whl вручную и положить в каталог проекта:
#    https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
#    (выбрать файл под свою архитектуру, например linux_aarch64)

# 3. Скачать модели детекции и распознавания
./download_models.sh

# 4. Добавить себя в базу
mkdir -p known_faces/Алексей
cp ~/фото.jpg known_faces/Алексей/photo.jpg

# 5. Запустить
source venv/bin/activate
python3 main.py
```

## CLI-аргументы

```bash
python3 main.py                                    # всё по умолчанию
python3 main.py --port 9090 --threshold 0.50       # другой порт и порог
python3 main.py --lang en --no-tts                  # английский, без озвучки
python3 main.py --camera 2 --data-dir /mnt/data     # конкретная камера, другой каталог данных
python3 main.py --no-web --display                   # без веба, с окном OpenCV
python3 main.py --no-debug                           # без debug-вывода в консоль
```

Env-переменные (задают умолчания, CLI имеет приоритет):
`FACE_PORT`, `FACE_THRESHOLD`, `FACE_LANG`, `FACE_CAMERA`, `FACE_DATA_DIR`

## Веб-интерфейс

В браузере: `http://<ip-orange-pi>:8080`

Показывает живой кадр с камеры (~3 fps) и последние 15 событий с миниатюрами. Кадры отдаются из памяти (без записи на диск). Кнопка `↻` перезагружает базу лиц без рестарта.

### Эндпоинты

| Endpoint | Описание |
|---|---|
| `GET /` | Веб-UI |
| `GET /frame.jpg` | Живой JPEG-кадр (из памяти) |
| `GET /detections.json` | Последние события (JSON) |
| `GET /snap/<id>.jpg` | Снапшот события (BLOB из SQLite) |
| `GET /health` | Статус: `{uptime_s, last_detection_ts, frame_jpeg_bytes}` |
| `GET /reload` | Перезагрузить базу лиц (hot-reload) |
| `GET /clear` | Удалить все события и снапшоты из БД |
| `GET /debug/aligned.jpg` | Последнее выровненное лицо (дебаг) |

## Добавление новых людей

Создай папку с именем человека и положи туда одно или несколько фото (лицо хорошо видно):

```
known_faces/
  Алексей/
    photo.jpg
  Мария/
    photo.jpg
    photo2.jpg
```

База перезагружается автоматически каждые 30 секунд при изменении файлов, либо по кнопке `↻` в веб-UI, либо через `GET /reload`.

## Настройки

| Параметр | По умолчанию | CLI | Назначение |
|----------|--------------|-----|------------|
| `recognition_threshold` | 0.45 | `--threshold` | Порог косинусного сходства (выше — строже) |
| `stranger_min_score` | 0.30 | — | Ниже — не считаем незнакомцем (отсечка ложных) |
| `greet_cooldown` | 10 с | — | Пауза между приветствиями одного человека |
| `confirm_frames` | 3 | — | Кадров подряд для подтверждения |
| `score_window` | 7 | — | Окно сглаживания скора |
| `web_event_cooldown` | 30 с | — | Минимальный интервал между одинаковыми событиями |
| `stranger_confirm_delay` | 5 с | — | Задержка перед записью незнакомца |
| `lang` | ru | `--lang` | Язык приветствий (ru / en) |
| `web_port` | 8080 | `--port` | Порт веб-интерфейса (0 = выключен) |

## Автозапуск при загрузке системы

На Orange Pi под пользователем `orangepi`:

```bash
# 1. Скопировать unit в пользовательский systemd
mkdir -p ~/.config/systemd/user
cp /path/to/faces/faces.service ~/.config/systemd/user/

# 2. Включить запуск без входа в систему (один раз!)
sudo loginctl enable-linger orangepi

# 3. Включить и запустить сервис
systemctl --user daemon-reload
systemctl --user enable faces
systemctl --user start faces
```

Полезные команды:

- **Логи**: `tail -f ~/faces/faces.log`
- **Перезапустить**: `systemctl --user restart faces`
- **Остановить**: `systemctl --user stop faces`
- **Отключить автозапуск**: `systemctl --user disable faces`

### Если после перезагрузки сервис не запустился

1. **Linger**: `loginctl show-user $USER | grep Linger` — должно быть `Linger=yes`
2. **Enabled**: `systemctl --user is-enabled faces` — должно быть `enabled`
3. **Статус**: `systemctl --user status faces` + `tail -n 80 ~/faces/faces.log`
