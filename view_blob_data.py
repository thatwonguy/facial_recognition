import sqlite3
import cv2
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect("faces.db")
c = conn.cursor()


# Function to retrieve and display BLOB data
def view_blob_data():
    # Retrieve BLOB data from the database
    c.execute("SELECT * FROM faces")
    rows = c.fetchall()

    for row in rows:
        # Extract timestamp and BLOB data
        timestamp = row[1]
        face_blob = row[2]

        # Decode BLOB data
        nparr = np.frombuffer(face_blob, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Display the image
        cv2.imshow("Face", img)
        cv2.waitKey(0)  # Press any key to continue viewing images

    # Close the database connection
    conn.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    view_blob_data()
