import cv2
import face_recognition

img1 = face_recognition.load_image_file('obama.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1Test = face_recognition.load_image_file('obama2.jpg')
img1Test = cv2.cvtColor(img1Test, cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(img1)[0]
print(face)
encodeFace = face_recognition.face_encodings(img1)[0]
print(encodeFace)
cv2.rectangle(img1, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)

faceTest = face_recognition.face_locations(img1Test)[0]
encodeTestFace = face_recognition.face_encodings(img1Test)[0]
cv2.rectangle(img1Test, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeFace], encodeTestFace)
faceDis = face_recognition.face_distance([encodeFace], encodeTestFace)
print(results, faceDis)
cv2.putText(img1Test, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Obama', img1)
cv2.imshow('Obama Test', img1Test)
cv2.waitKey(0)
cv2.destroyAllWindows()