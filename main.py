#!/usr/bin/env python3
"""
Распознавание лиц на входе в квартиру.
Добавление человека: положи фото в known_faces/ИМЯ/photo.jpg
"""

import os
import time
import subprocess
import threading
import cv2
import numpy as np
import face_recognition

# --- Настройки ---
KNOWN_FACES_DIR = "known_faces"
CAMERA_INDEX = 0          # /dev/video0, попробуй 1 или 2 если не работает
FRAME_SCALE = 0.5         # уменьшаем кадр для скорости (0.5 = половина)
RECOGNITION_TOLERANCE = 0.55  # меньше = строже (0.4–0.6)
GREET_COOLDOWN = 300      # секунд между приветствиями одного человека
PROCESS_EVERY_N = 3       # обрабатывать каждый N-й кадр

GREETINGS = {
    "ru": "Привет, {}!",
    "en": "Hello, {}!",
}
LANG = "ru"

UNKNOWN_LABEL = "Незнакомец"

# -------------------------


def load_known_faces(directory: str):
    encodings = []
    names = []

    if not os.path.isdir(directory):
        print(f"[!] Папка {directory} не найдена. Создай её и добавь фото.")
        return encodings, names

    for person_name in sorted(os.listdir(directory)):
        person_dir = os.path.join(directory, person_name)
        if not os.path.isdir(person_dir):
            continue

        photos_loaded = 0
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)

            if not face_encs:
                print(f"[!] Лицо не найдено: {path}")
                continue

            encodings.append(face_encs[0])
            names.append(person_name)
            photos_loaded += 1

        if photos_loaded > 0:
            print(f"[+] Загружено {photos_loaded} фото: {person_name}")
        else:
            print(f"[!] Нет подходящих фото в {person_dir}")

    print(f"[*] Всего в базе: {len(set(names))} человек(а)")
    return encodings, names


def speak(text: str):
    """Произносит текст через espeak-ng в отдельном потоке."""
    def _run():
        try:
            if LANG == "ru":
                subprocess.run(
                    ["espeak-ng", "-v", "ru", "-s", "140", text],
                    check=True, capture_output=True
                )
            else:
                subprocess.run(
                    ["espeak-ng", text],
                    check=True, capture_output=True
                )
        except FileNotFoundError:
            print(f"[!] espeak-ng не установлен. Приветствие: {text}")
        except subprocess.CalledProcessError as e:
            print(f"[!] Ошибка TTS: {e}")

    threading.Thread(target=_run, daemon=True).start()


def identify_face(face_encoding, known_encodings, known_names):
    if not known_encodings:
        return UNKNOWN_LABEL

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = int(np.argmin(distances))

    if distances[best_idx] <= RECOGNITION_TOLERANCE:
        return known_names[best_idx]
    return UNKNOWN_LABEL


def draw_box(frame, top, right, bottom, left, name, color):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, name,
        (left + 6, bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )


def main():
    print("[*] Загрузка базы лиц...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)

    print(f"[*] Открытие камеры (индекс {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] Не удаётся открыть камеру {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[*] Запуск. Нажми Q для выхода.")

    last_greeted: dict[str, float] = {}
    frame_count = 0

    # Результаты последнего анализа
    face_locations = []
    face_names = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Нет кадра с камеры")
            time.sleep(0.1)
            continue

        frame_count += 1

        # Анализируем не каждый кадр — экономим CPU
        if frame_count % PROCESS_EVERY_N == 0:
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_names = []
            for enc in face_encodings:
                name = identify_face(enc, known_encodings, known_names)
                face_names.append(name)

                if name != UNKNOWN_LABEL:
                    now = time.time()
                    last_time = last_greeted.get(name, 0)
                    if now - last_time > GREET_COOLDOWN:
                        last_greeted[name] = now
                        greeting = GREETINGS[LANG].format(name)
                        print(f"[>] {greeting}")
                        speak(greeting)

        # Рисуем рамки (масштабируем координаты обратно)
        scale = 1.0 / FRAME_SCALE
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top = int(top * scale)
            right = int(right * scale)
            bottom = int(bottom * scale)
            left = int(left * scale)

            color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
            draw_box(frame, top, right, bottom, left, name, color)

        cv2.imshow("Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Завершено.")


if __name__ == "__main__":
    main()
