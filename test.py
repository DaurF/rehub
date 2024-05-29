import tensorflow as tf
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
from helpers import calculate_body_angles, get_visible_side
import model as algos

# Загрузка моделей
modelYolo = YOLO('yolov8s.pt')
model = tf.keras.models.load_model('exercise2.keras')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

message = 'Start doing exercise!'

# video_path = 'input_video_from_Maulen/1.mp4'
video_path = 'input/correct_1_m.MOV'

cap = cv2.VideoCapture(video_path)


def put_message():
    color = (0, 255, 0) if message == 'Good job!' else (0, 0, 255)
    if 'might' in message.lower():
        color = (0, 255, 255)


    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3,
                cv2.LINE_4)


if not cap.isOpened():
    print("Error opening video stream or file")
cnt = 0
while cap.isOpened():
    cnt += 1
    ret, frame = cap.read()
    if not ret:
        break
    if cnt % 3 != 0:
        continue
    # Использование YOLO для обнаружения людей

    put_message()

    results = modelYolo(frame)
    detections = results[0].boxes.data

    for detection in detections:
        # Определяем координаты ограничивающей рамки и класс обнаруженного объекта
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) == 0:  # Проверяем, что обнаружен человек (класс 0 для человека в COCO датасете)
            # Обрезка кадра по координатам рамки
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

            # Предобработка обрезанного кадра
            resized_frame = cv2.resize(cropped_frame, (224, 224))  # Изменяем размер до 224x224 пикселей
            normalized_frame = resized_frame / 255.0  # Нормализуем значения пикселей
            input_frame = np.expand_dims(normalized_frame, axis=0)  # Добавляем ось для batch

            # Предсказание
            predictions = model.predict(input_frame)
            predicted_class = np.argmax(predictions, axis=1)[0]

            print(predictions)

            if predicted_class == 0:
                message = 'Start doing exercise!'
            else:
                result = pose.process(cropped_frame)

                # Draw the pose landmarks on the frame
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark

                    data = calculate_body_angles(landmarks)
                    data['visible_side'] = get_visible_side(landmarks)

                    if predicted_class == 1:
                        try:
                            algos.check_exercise_1(data, landmarks)
                            message = 'Good job!'
                        except RuntimeError as err:
                            message = err.args[0]

                    if predicted_class == 2:
                        try:
                            algos.check_exercise_2(data, landmarks)
                            message = 'Good job!'
                        except RuntimeError as err:
                            print(err)
                            print(f"Right elbow: {data['left_elbow']}; Right shoulder: {data['left_shoulder']}")
                            message = err.args[0]

            # print(predictions)
            # Определение класса упражнения
            # class_labels = {2: 'Background', 0: 'First_Exercise', 1: 'Second_exercise', 3: 'Third_exercise'}  # Замените метки на свои классы
            # label = class_labels[predicted_class]

            # Отображение метки на кадре
            # cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Отображение рамки вокруг человека
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Показ кадра
    cv2.imshow('Exercise Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
