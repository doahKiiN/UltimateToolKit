# UltimateToolKit.py

import cv2
from ffpyplayer.player import MediaPlayer
import tensorflow as tf
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import socket

class Video:
    def __init__(self, file_path):
        """
        Инициализация объекта Video с указанием пути к видеофайлу.
        """
        self.file_path = file_path
        self.cap = cv2.VideoCapture(self.file_path)
        self.player = MediaPlayer(self.file_path)
        self.is_playing = False
        self.is_paused = False

    def play(self):
        """
        Воспроизведение видео.
        """
        if not self.cap.isOpened():
            print(f"Ошибка: Не удалось открыть файл {self.file_path}")
            return

        self.is_playing = True
        while self.cap.isOpened() and self.is_playing:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                cv2.imshow('Video Player', frame)

                # Обработка аудио
                audio_frame, val = self.player.get_frame()
                if val != 'eof' and audio_frame is not None:
                    img, t = audio_frame

            # Обработка событий окна
            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                self.stop()
            elif key & 0xFF == ord('p'):
                self.pause()

            # Проверка, было ли окно закрыто пользователем
            if cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def pause(self):
        """
        Пауза воспроизведения.
        """
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("Пауза")
        else:
            print("Продолжение воспроизведения")

    def stop(self):
        """
        Остановка воспроизведения.
        """
        self.is_playing = False
        self.is_paused = False
        print("Остановка воспроизведения")

    def __del__(self):
        """
        Освобождение ресурсов при удалении объекта.
        """
        if self.cap.isOpened():
            self.cap.release()
        self.player.close_player()

# Внутренний модуль mp4play
class mp4play:
    Video = Video

# Класс для работы с фотографиями
class Photo:
    def __init__(self, file_path):
        """
        Инициализация объекта Photo с указанием пути к фотографии.
        """
        self.file_path = file_path
        self.image = cv2.imread(self.file_path)

    def open(self):
        """
        Открытие и отображение фотографии.
        """
        if self.image is None:
            print(f"Ошибка: Не удалось открыть файл {self.file_path}")
            return

        cv2.imshow('Photo Viewer', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def resize(self, width, height):
        """
        Изменение размера фотографии.
        """
        self.image = cv2.resize(self.image, (width, height))

    def grayscale(self):
        """
        Преобразование фотографии в оттенки серого.
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def edges(self):
        """
        Обнаружение границ на фотографии.
        """
        self.image = cv2.Canny(self.image, 100, 200)

    def blur(self, kernel_size=(5, 5)):
        """
        Размытие фотографии.
        """
        self.image = cv2.GaussianBlur(self.image, kernel_size, 0)

    def sharpen(self):
        """
        Усиление резкости фотографии.
        """
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.image = cv2.filter2D(self.image, -1, kernel)

    def threshold(self, threshold_value=127, max_value=255):
        """
        Бинаризация фотографии.
        """
        _, self.image = cv2.threshold(self.image, threshold_value, max_value, cv2.THRESH_BINARY)

    def detect_faces(self, cascade_path):
        """
        Обнаружение лиц на фотографии.
        """
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def detect_objects(self, model_path):
        """
        Обнаружение объектов на фотографии с использованием предварительно обученной модели (например, SSD).
        """
        net = cv2.dnn.readNetFromCaffe(model_path)
        blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([self.image.shape[1], self.image.shape[0], self.image.shape[1], self.image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(self.image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    def save(self, output_path):
        """
        Сохранение фотографии в файл.
        """
        cv2.imwrite(output_path, self.image)

# Внутренний модуль photo
class photo:
    Photo = Photo

# Класс для работы с TensorFlow моделями
class TensorFlowModel:
    def __init__(self, model_path):
        """
        Инициализация объекта TensorFlowModel с указанием пути к модели.
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, input_data):
        """
        Выполнение предсказания с использованием загруженной модели.
        """
        return self.model.predict(input_data)

# Внутренний модуль tensorflow_model
class tensorflow_model:
    TensorFlowModel = TensorFlowModel

# Класс для работы с файловой системой
class FileSystem:
    @staticmethod
    def check_file_exists(file_path):
        """
        Проверка существования файла.
        """
        return os.path.exists(file_path)

    @staticmethod
    def create_directory(directory_path):
        """
        Создание директории.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    @staticmethod
    def delete_file(file_path):
        """
        Удаление файла.
        """
        if os.path.exists(file_path):
            os.remove(file_path)

# Внутренний модуль file_system
class file_system:
    FileSystem = FileSystem

# Класс для работы с графическим интерфейсом пользователя (GUI)
class TkinterApp:
    def __init__(self, root):
        """
        Инициализация объекта TkinterApp.
        """
        self.root = root
        self.root.title("UltimateToolKit GUI")

        # Кнопка для открытия видео
        self.open_video_button = tk.Button(root, text="Открыть видео", command=self.open_video)
        self.open_video_button.pack()

        # Кнопка для открытия фотографии
        self.open_photo_button = tk.Button(root, text="Открыть фото", command=self.open_photo)
        self.open_photo_button.pack()

    def open_video(self):
        """
        Открытие видеофайла.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            video = mp4play.Video(file_path)
            video.play()

    def open_photo(self):
        """
        Открытие фотографии.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if file_path:
            photo = photo.Photo(file_path)
            photo.open()

# Внутренний модуль tkinter_app
class tkinter_app:
    TkinterApp = TkinterApp

# Класс для работы с сетевыми сокетами (клиент)
class SocketClient:
    def __init__(self, host, port):
        """
        Инициализация объекта SocketClient.
        """
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        """
        Подключение к серверу.
        """
        self.client_socket.connect((self.host, self.port))

    def send_message(self, message):
        """
        Отправка сообщения на сервер.
        """
        self.client_socket.sendall(message.encode())

    def receive_message(self, buffer_size=1024):
        """
        Получение сообщения от сервера.
        """
        data = self.client_socket.recv(buffer_size)
        return data.decode()

    def close(self):
        """
        Закрытие соединения.
        """
        self.client_socket.close()

# Класс для работы с сетевыми сокетами (сервер)
class SocketServer:
    def __init__(self, host, port):
        """
        Инициализация объекта SocketServer.
        """
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

    def accept_connection(self):
        """
        Принятие входящего соединения.
        """
        self.client_socket, self.client_address = self.server_socket.accept()
        print(f"Подключение от {self.client_address}")

    def receive_message(self, buffer_size=1024):
        """
        Получение сообщения от клиента.
        """
        data = self.client_socket.recv(buffer_size)
        return data.decode()

    def send_message(self, message):
        """
        Отправка сообщения клиенту.
        """
        self.client_socket.sendall(message.encode())

    def close(self):
        """
        Закрытие соединения.
        """
        self.client_socket.close()
        self.server_socket.close()

# Внутренний модуль socket_client
class socket_client:
    SocketClient = SocketClient

# Внутренний модуль socket_server
class socket_server:
    SocketServer = SocketServer