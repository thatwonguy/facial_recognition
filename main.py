import cv2
import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect("faces.db")
c = conn.cursor()

# Create table to store faces if not exists
c.execute(
    """CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
             timestamp TIMESTAMP,
             face BLOB)"""
)

# Initialize face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Variables for timer
capture_interval = 5  # in seconds
last_capture_time = datetime.now()


# Function to capture and save faces
def capture_face(camera_index=0):
    global last_capture_time

    # Open camera
    cap = cv2.VideoCapture(camera_index)

    while True:
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for x, y, w, h in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop face
            face = frame[y : y + h, x : x + w]

            # Get current timestamp
            timestamp = datetime.now()

            # Check if 5 seconds have passed since last capture
            if (timestamp - last_capture_time).total_seconds() >= capture_interval:
                # Save face and timestamp to database
                c.execute(
                    "INSERT INTO faces (timestamp, face) VALUES (?, ?)",
                    (timestamp, cv2.imencode(".jpg", face)[1].tobytes()),
                )
                conn.commit()
                # Update last capture time
                last_capture_time = timestamp

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        print("Pressed key:", key)

        # Press 'Esc' to exit
        if key == 27:  # ASCII code for 'Esc' key
            break

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_face()
