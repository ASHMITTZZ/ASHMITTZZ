import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

Ashmit_image = face_recognition.load_image_file("faces/Ashmit.jpg")
Ashmit_encoding = face_recognition.face_encodings(Ashmit_image)[0]

rohan_image = face_recognition.load_image_file("faces/rohan.jpg")
rohan_encoding = face_recognition.face_encodings(rohan_image)[0]

known_face_encodings = [Ashmit_encoding, rohan_encoding]
known_face_names = ["Ashmit", "Rohan"]
# list of expected students

students = known_face_names.copy()
face_location = []
face_encoding = []
# get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            
            name = known_face_names[best_match_index]

            if name in known_face_names:
                font=cv2.FONT_HERSHEY_COMPLEX
                bottomleftcorneroftext=(10,100)
                fontscale= 1.5
                fontcolor=(255,0,0)
                thickness=3
                linetype=2
                cv2.puttext(frame,name("present"),bottomleftcorneroftext,fontscale,fontcolor,thickness,linetype,font)
              
            if name in students:
                students.remove(name)  # Mark attendance for the recognized student
                # Write the attendance information to the CSV file
                lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S")])

        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow("Attendance", frame)

 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close the CSV file
video_capture.release()
f.close()
cv2.destroyAllWindows()
