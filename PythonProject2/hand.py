import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Start webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)  # modifies img with hand landmarks
    totalFingers = 0

    if hands:
        for hand in hands:
            fingers = detector.fingersUp(hand)
            totalFingers += fingers.count(1)

    # ----- Create Second Window Image (Only Finger Count) -----
    finger_display = np.zeros((300, 400, 3), dtype=np.uint8)  # black background
    cv2.putText(finger_display, f'Total Fingers: {totalFingers}', (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # ----- Show Both Windows -----
    cv2.imshow("Live Webcam Feed", img)  # Webcam + Hands
    cv2.imshow("Fingers Count", finger_display)  # Only Count Display

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
