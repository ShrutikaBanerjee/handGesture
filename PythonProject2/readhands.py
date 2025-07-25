import cv2
from cvzone.HandTrackingModule import HandDetector

# Start webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    totalFingers = 0  # Initialize total finger count

    if hands:
        for hand in hands:
            fingers = detector.fingersUp(hand)
            totalFingers += fingers.count(1)  # Count number of fingers up per hand

        print(f"Total Fingers Up: {totalFingers}")  # Print combined total

    cv2.imshow("Hand Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
