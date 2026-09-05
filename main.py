import cv2
import os
from deepface import DeepFace
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def detect_faces_and_emotions():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở Webcam.")
        return

    print("Đang khởi động Vision AI với phân tích biểu cảm...")
    print("Lưu ý: Lần chạy đầu tiên sẽ mất thời gian tải dữ liệu mô hình (khoảng vài trăm MB).")
    print("Nhấn 'q' để thoát.")

    frame_count = 0
    last_emotion = "Đang phân tích..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if frame_count % 10 == 0:
                try:
                    face_roi = frame[y:y+h, x:x+w]
    
                    results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if results:
                        last_emotion = results[0]['dominant_emotion']
                        
                        emotion_vn = {
                            'angry': 'Gian du',
                            'disgust': 'Ghe tom',
                            'fear': 'So hai',
                            'happy': 'Vui ve',
                            'sad': 'Buon',
                            'surprise': 'Ngac nhien',
                            'neutral': 'Binh thuong'
                        }
                        last_emotion = emotion_vn.get(last_emotion, last_emotion)
                except Exception as e:
                    print(f"Lỗi phân tích: {e}")

            cv2.putText(frame, f"Emotion: {last_emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Vision AI - Face & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_and_emotions()
