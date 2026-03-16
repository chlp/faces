# faces

Приложение для Orange Pi 5 Max (RK3588): камера смотрит на вход, узнаёт людей по лицу и приветствует их по имени через espeak-ng. Детекция и распознавание — на NPU (модели SCRFD + ArcFace в формате RKNN).

## Что внутри

- `main.py` — основное приложение (один файл, вся логика)
- `known_faces/` — база лиц: одна папка на человека, внутри — фото (`.jpg` / `.png`)
- `models/` — RKNN-модели (скачиваются скриптом `download_models.sh`)
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
#    Если .whl уже лежит рядом с install.sh — он подхватится при установке.

# 3. Скачать модели детекции и распознавания
./download_models.sh

# 4. Добавить себя в базу
mkdir -p known_faces/Алексей
cp ~/фото.jpg known_faces/Алексей/photo.jpg

# 5. Запустить
source venv/bin/activate
python3 main.py
```

Веб-интерфейс поднимается самим `main.py` — отдельно ничего запускать не нужно.

## Веб-интерфейс

В браузере: `http://<ip-orange-pi>:8080`

Показывает текущий кадр с камеры (обновление раз в секунду) и последние 15 обнаружений с миниатюрами в момент детекции.

Данные лежат в `/tmp/faces_web/` и пересоздаются при каждом запуске.

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

После добавления или смены фото приложение нужно перезапустить.

## Настройки (в начале main.py)

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `RECOGNITION_THRESHOLD` | 0.55 | Порог косинусного сходства для «узнал человека» (выше — строже). |
| `STRANGER_MIN_SCORE` | 0.20 | Ниже этого сходства детекция не считается «незнакомцем» (отсекаются ложные срабатывания). |
| `GREET_COOLDOWN` | 10 | Секунд между приветствиями одного и того же человека. |
| `CONFIRM_FRAMES` | 3 | Сколько кадров подряд нужно видеть человека перед приветствием. |
| `WEB_PORT` | 8080 | Порт веб-интерфейса; 0 — выключен. |

## Автозапуск при загрузке системы

На Orange Pi под пользователем `orangepi`:

```bash
# 1. Скопировать unit в пользовательский systemd
mkdir -p ~/.config/systemd/user
cp /path/to/faces/faces.service ~/.config/systemd/user/

# 2. Включить запуск без входа в систему (один раз!)
# Важно: эту команду нужно выполнить с root (sudo)
sudo loginctl enable-linger orangepi

# 3. Включить и запустить сервис
systemctl --user daemon-reload
systemctl --user enable faces
systemctl --user start faces
```

Полезные команды:

- **Логи** — сервис пишет в файл `faces.log` в каталоге проекта (на Orange Pi user journal часто недоступен, поэтому логи в файле):
  - следить в реальном времени: `tail -f ~/faces/faces.log`
  - последние 200 строк: `tail -n 200 ~/faces/faces.log`
  - с прокруткой (выход из follow: Ctrl+C, снова: F): `tail -f ~/faces/faces.log | less +F`
- Перезапустить: `systemctl --user restart faces`
- Остановить: `systemctl --user stop faces`
- Отключить автозапуск: `systemctl --user disable faces`

### Если после перезагрузки сервис не запустился

Зайди на Orange Pi **под пользователем orangepi** (под которым ставил сервис) и выполни по порядку:

**1. Включён ли linger (без него user-сервисы при загрузке не стартуют):**
```bash
loginctl show-user $USER | grep Linger
```
Должно быть `Linger=yes`. Если `no` — один раз с root: `sudo loginctl enable-linger orangepi`, затем перезагрузка.

**2. Включён ли сервис:**
```bash
systemctl --user is-enabled faces
```
Должно быть `enabled`. Если `disabled` — снова: `systemctl --user enable faces`.

**3. Состояние сервиса и последние логи:**
```bash
systemctl --user status faces
tail -n 80 ~/faces/faces.log
```
По статусу видно: active, failed или inactive. В логах — причина (камера, NPU, путь к venv и т.д.).

**4. Запускался ли вообще user manager после загрузки:**
```bash
systemctl --user status
```
Если сервис inactive/failed — смотреть вывод `status` и конец `faces.log`.
