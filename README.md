# faces

Приложение для Orange Pi 5 Max: камера смотрит на вход в квартиру, узнаёт людей и приветствует их по имени.

## Что внутри

- `main.py` — основное приложение
- `known_faces/` — база лиц (одна папка = один человек)
- `install.sh` — установка всех зависимостей на Orange Pi
- `requirements.txt` — Python-пакеты

## Быстрый старт

```bash
# 1. Установить зависимости
chmod +x install.sh && ./install.sh

# 2. Добавить себя в базу
mkdir -p known_faces/Алексей
cp ~/фото.jpg known_faces/Алексей/photo.jpg

# 3. Запустить
source venv/bin/activate
python3 main.py
```

`main.py` автоматически запускает дочерний процесс `python3 -m http.server` для веб-интерфейса — вручную ничего дополнительно запускать не нужно.

## Веб-интерфейс

Откройте в браузере: `http://<ip-orange-pi>:8080`

Показывает текущий кадр с камеры (обновляется раз в секунду) и последние 10 обнаружений с миниатюрами кадров в момент детекции.

Файлы хранятся в `/tmp/faces_web/` и пересоздаются при каждом запуске.

## Добавление новых людей

Создай папку с именем человека и положи туда одно или несколько фото (`.jpg` / `.png`):

```
known_faces/
  Алексей/
    photo.jpg
  Мария/
    photo.jpg
    photo2.jpg
```

Перезапускать приложение после добавления фото — обязательно.

## Автозапуск при загрузке системы

На Orange Pi под пользователем `orangepi`:

```bash
# 1. Скопировать unit в пользовательский systemd
mkdir -p ~/.config/systemd/user
cp /home/orangepi/faces/faces.service ~/.config/systemd/user/

# 2. Включить запуск без входа в систему (один раз)
loginctl enable-linger orangepi

# 3. Включить и запустить сервис
systemctl --user daemon-reload
systemctl --user enable faces
systemctl --user start faces
```

Полезные команды:

- Логи в реальном времени: `journalctl --user -u faces -f`
- Остановить: `systemctl --user stop faces`
- Отключить автозапуск: `systemctl --user disable faces`
