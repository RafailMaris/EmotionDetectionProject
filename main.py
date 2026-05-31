import cv2
from fer.fer import FER
import numpy as np
import math_functions
import classifier
#from deepface import DeepFace
def on_still_image():#demo

    img_demo = cv2.imread("C:\\Users\\Legion\\Downloads\\0_8OuVyFarJa0lVIRt.jpg")

    if img_demo is None:
        print("Image not found!")
        exit()

    img_demo = cv2.flip(img_demo, 1)

    # gray
    gray_demo = cv2.cvtColor(img_demo, cv2.COLOR_BGR2GRAY)

    pq = classifier.naive_viola_and_jones(gray_demo, 1.25)

    #aplicam NMS
    final_faces = math_functions.non_maxima_suppression(pq, iou_threshold=0.5)

    #desenam ce ramane
    output_img = cv2.cvtColor(gray_demo, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in final_faces:
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_img, "Face", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Final Detections", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#classifier ul antrenat

emotion_detector = FER(mtcnn=False)  # mtcnn=True is more accurate but slower -> multi-task cascade convolutional neural network
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def draw_emotion(face_roi, img, x, y):

    #face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0:
        return

    result = emotion_detector.detect_emotions(face_roi)

    if result:
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get) # most likely emotion
        confidence = emotions[top_emotion] #percentage

        label = f"{top_emotion} ({confidence:.0%})"
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

def analyze_camera(cap,GAMMA_LUT, face_cascade, eye_detector):

    _, img = cap.read()
    #preprocessing
    img = cv2.flip(img, 1)  # camera da imaginea flipped
    img = cv2.LUT(img, GAMMA_LUT)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# gray = math_functions.gamma_correction(gray)

    #gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, 1.25, 4)# aici detecteaza efectiv modeluț
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + h
        y2 = y + h
        math_functions.place_rectangles_in_image(img, y1, x1, y2, x2)

        #roi tightenȘ
        # x1 = max(x + int(0.05 * w), 0)
        # y1 = max(y + int(0.05 * h), 0)
        # x2 = min(x + w - int(0.05 * w), img.shape[1])
        # y2 = min(y + h - int(0.05 * h), img.shape[0])
        # face_roi = img[y1:y2, x1:x2]

        face_roi = img[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        #face_roi = math_functions.align_face(img, face_roi)

        face_roi = cv2.resize(face_roi, (224, 224))
        #face_roi = math_functions.Face_Alignment(face_roi)
        face_roi = math_functions.align_face(face_roi, eye_detector)
        draw_emotion(face_roi,img,x1,y1)

    cv2.imshow('img', img)


if __name__ == "__main__":
    eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # luam de la camera
    GAMMA_LUT = math_functions.build_gamma_lut(gamma=1.2)
    while cap.isOpened():

        analyze_camera(cap,GAMMA_LUT,face_cascade, eye_detector)

        key = cv2.waitKey(1) & 0xFF  # daca apasam q: stop. daca apasam a: demo
        if key == ord("q"):
            break

        if key == ord("a"):
            on_still_image()

    cap.release()



