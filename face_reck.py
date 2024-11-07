import cv2
import face_recognition
import os
import numpy as np

# Load known images and encode them
def load_and_encode_images(directory):
    known_face_encodings = []
    known_face_names = []
    
    # Loop through each image in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load image
            image = face_recognition.load_image_file(os.path.join(directory, filename))
            # Get the face encoding
            face_encoding = face_recognition.face_encodings(image)[0]
            # Split the filename at the underscore and take the first part (before the number)
            name = os.path.splitext(filename)[0].split('_')[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
    
    return known_face_encodings, known_face_names

# Load your own face images from the folder "images_of_me"
known_face_encodings, known_face_names = load_and_encode_images("images_of_me")

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

# Get the frame width and height from the video capture
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Define the codec and create a VideoWriter object to save the video to an MP4 file
# 'XVID' codec is common for MP4 format, but 'mp4v' might also work depending on the system
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))  # 20 FPS

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (OpenCV default) to RGB (face_recognition uses RGB)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up face locations since the frame was resized to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Draw a square (rectangle with equal sides) around the face
        face_width = right - left
        face_height = bottom - top
        face_size = min(face_width, face_height)
        center_x = left + face_width // 2
        center_y = top + face_height // 2

        half_size = face_size // 2
        square_left = center_x - half_size
        square_top = center_y - half_size
        square_right = center_x + half_size
        square_bottom = center_y + half_size

        # Draw the square
        cv2.rectangle(frame, (square_left, square_top), (square_right, square_bottom), (0, 255, 0), 2)
        
        # Draw the name inside the square
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(name, font, 0.5, 1)[0]
        text_x = square_left + (face_size - text_size[0]) // 2
        text_y = square_top + (face_size + text_size[1]) // 2

        # Draw the name inside the square
        cv2.putText(frame, name, (text_x, text_y), font, 0.5, (255, 255, 255), 1)
    
    # Write the frame with the rectangle and name to the output video file
    out.write(frame)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and the video writer
video_capture.release()
out.release()  # Save the video file
cv2.destroyAllWindows()
