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


# Function to capture and save faces
def capture_face(camera_index=0):
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    # cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

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

            # Save face and timestamp to database
            c.execute(
                "INSERT INTO faces (timestamp, face) VALUES (?, ?)",
                (timestamp, cv2.imencode(".jpg", face)[1].tobytes()),
            )
            conn.commit()

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release camera and close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_face()
