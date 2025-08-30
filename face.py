import cv2
import csv
import os

# Function to register faces and save names to CSV
def register_faces():
    try:
        # Load pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Error loading face cascade classifier")

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Error: Could not open video capture")

        # Initialize an empty list to store face coordinates and names
        face_data = []
        name = ""

        print("Press 'c' to capture a face and enter a name")
        print("Press 'q' to quit")

        # Start capturing video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the frame
            cv2.imshow('Face Registration - Press c to capture, q to quit', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                name = input("Enter the name of the person: ")
                x, y, w, h = faces[0]  # Take the first detected face
                face_data.append((name, (x, y, w, h)))
                print(f"Face registered for {name}")
            elif key == ord('q'):
                break

        # Release video capture
        cap.release()
        cv2.destroyAllWindows()

        # Save face data to CSV
        csv_path = os.path.join(os.path.dirname(__file__), 'attendance.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Name', 'X', 'Y', 'Width', 'Height'])
            for data in face_data:
                name, (x, y, w, h) = data
                writer.writerow([name, x, y, w, h])
        
        print(f"Face data saved to {csv_path}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        return False
    
    return True

# Main function
if __name__ == "__main__":
    register_faces()