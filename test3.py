import face_recognition
import numpy as np

def best_value(face_encodings, known_face_encodings):
    A = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        A.append(face_distances)
    return min(A)


minju_image = face_recognition.load_image_file("./known/minju.jpg")
minju_face_encoding = face_recognition.face_encodings(minju_image)[0]
print(minju_image)


known_face_encodings = [
    minju_face_encoding,
]
known_face_names = [
    "Min-Ju"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("./unknown/unknown.jpg")
#cnn 개느림
face_locations = face_recognition.face_locations(unknown_image)

face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

best = best_value(face_encodings, known_face_encodings)
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    print(face_distances)
    #하나만 찾는데에는 쓸데없음
    best_match_index = np.argmin(face_distances)
    if face_distances == best:
        name = known_face_names[best_match_index]
        print(top, right, bottom, left)
#445 1619 534 1529

print(best_value(face_encodings, known_face_encodings))
