import cv2
import os
from deepface import DeepFace
import logging

# Tắt bớt log của TensorFlow để đỡ rối
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def detect_faces_and_emotions():
    # Tải bộ phân loại khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Khởi tạo Webcam
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
        
        # Chuyển sang màu xám để phát hiện khuôn mặt nhanh hơn
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Vẽ khung khuôn mặt
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Chỉ phân tích biểu cảm mỗi 10 khung hình để tránh bị lag
            if frame_count % 10 == 0:
                try:
                    # Cắt vùng khuôn mặt để phân tích
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Phân tích cảm xúc sử dụng DeepFace
                    # enforce_detection=False giúp bỏ qua lỗi nếu không tìm thấy mặt trong ROI
                    results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                    if results:
                        # Lấy cảm xúc chính (dominant_emotion)
                        last_emotion = results[0]['dominant_emotion']
                        
                        # Dịch sang tiếng Việt cho thân thiện
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

            # Hiển thị biểu cảm lên khung hình
            cv2.putText(frame, f"Emotion: {last_emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị kết quả
        cv2.imshow('Vision AI - Face & Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_and_emotions()
