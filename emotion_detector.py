import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    print("Frame read:", ret)
    if not ret:
        print("Failed to grab frame")
        break

    try:
        # Analyze the frame using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        print("DeepFace result:", result)

        # Get emotion label
        emotion = result[0]['dominant_emotion']

        # Display the emotion on the screen
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Error in DeepFace analyze:", e)

    print("Showing frame...")
    # Show the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
