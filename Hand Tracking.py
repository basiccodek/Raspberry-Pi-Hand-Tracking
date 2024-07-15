import cv2
import mediapipe as mp
import RPi.GPIO as GPIO

# GPIOピン番号
LED_PIN = 18

# GPIOピンの初期設定
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # LEDを初期状態ではOFFにする

# MediaPipeの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # BGRからRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # フレームをMediaPipeに渡して手の検出を行う
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 指の数をカウント
            finger_count = 0
            for finger in [4, 8]:  # インデックスと中指の先端
                if hand_landmarks.landmark[finger].y < hand_landmarks.landmark[finger - 2].y:
                    finger_count += 1

            # 指が2本立てた場合にLEDを点灯
            if finger_count >= 2:
                GPIO.output(LED_PIN, GPIO.HIGH)  # LEDを点灯
            else:
                GPIO.output(LED_PIN, GPIO.LOW)  # LEDを消灯
            
            # 手の形を描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# GPIOを解放
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
