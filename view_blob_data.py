"""_summary_
Module that views data created by main.py script.
"""

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Face

# Create an SQLite engine and bind it to the Base
engine = create_engine("sqlite:///faces.db")

# Create a Session class bound to the engine
Session = sessionmaker(bind=engine)


# Function to retrieve and display BLOB data
def view_blob_data():
    """_summary_
    Main function that views data using sqlachemy.
    """
    # Create a session
    session = Session()

    # Retrieve all Face instances from the database
    faces = session.query(Face).all()

    for face in faces:
        # Extract timestamp and BLOB data
        timestamp = face.timestamp
        face_blob = face.face

        # Decode BLOB data
        nparr = np.frombuffer(face_blob, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Display the image
        cv2.imshow("Face", img)
        cv2.waitKey(0)  # Press any key to continue viewing images

    # Close the session
    session.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    view_blob_data()
