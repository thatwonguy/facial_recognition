import cv2
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the SQLAlchemy Base
Base = declarative_base()


# Define the Face class representing the table structure
class Face(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    face = Column(LargeBinary)


# Create an SQLite engine and bind it to the Base
engine = create_engine("sqlite:///faces.db")

# Create the database tables
Base.metadata.create_all(engine)

# Create a Session class bound to the engine
Session = sessionmaker(bind=engine)

# Initialize face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Variables for timer and first face flag
capture_interval = 5  # in seconds
last_capture_time = datetime.now()
first_face_detected = False


# Function to capture and save faces
def capture_face(camera_index=0):
    global last_capture_time, first_face_detected

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

            if not first_face_detected:
                # Create a new session
                session = Session()

                # Create a new Face instance and add it to the session
                new_face = Face(
                    timestamp=timestamp, face=cv2.imencode(".jpg", face)[1].tobytes()
                )
                session.add(new_face)

                # Commit the transaction
                session.commit()

                # Close the session
                session.close()

                # Set first_face_detected flag to True
                first_face_detected = True

            else:
                # Check if 5 seconds have passed since last capture
                if (timestamp - last_capture_time).total_seconds() >= capture_interval:
                    # Create a new session
                    session = Session()

                    # Create a new Face instance and add it to the session
                    new_face = Face(
                        timestamp=timestamp,
                        face=cv2.imencode(".jpg", face)[1].tobytes(),
                    )
                    session.add(new_face)

                    # Commit the transaction
                    session.commit()

                    # Close the session
                    session.close()

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
