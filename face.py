import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os

# Конфигурация путей к моделям
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ["Male", "Female"]
ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

color = (0, 255, 0)

# Загрузка нейросетей
faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False
    )
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                color,
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, faceBoxes


def process_frame(frame):
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    for faceBox in faceBoxes:
        face = frame[
            max(0, faceBox[1]) : min(faceBox[3], frame.shape[0] - 1),
            max(0, faceBox[0]) : min(faceBox[2], frame.shape[1] - 1),
        ]
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        cv2.putText(
            resultImg,
            f"{gender}, {age}",
            (faceBox[0], faceBox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return resultImg


def select_file():
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")]
    )
    if path:
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Ошибка", "Не удалось открыть изображение.")
            return
        result = process_frame(frame)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def use_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось открыть камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_frame(frame)
        cv2.imshow("Result", result)

        # Выход по нажатию клавиши ESC
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Графический интерфейс ---
root = tk.Tk()
root.title("Распознавание возраста и пола")
root.geometry("400x200")

label = tk.Label(
    root, text="Выберите изображение или используйте камеру:", font=("Arial", 14)
)
label.pack(pady=20)

btn_file = tk.Button(root, text="Выбрать файл", command=select_file, width=20, height=2)
btn_file.pack(pady=5)

btn_camera = tk.Button(
    root, text="Использовать камеру", command=use_camera, width=20, height=2
)
btn_camera.pack(pady=5)

root.mainloop()
