import os
import cv2
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

dataset = "dataset"
name = "NAME"

path = os.path.join(dataset, name)
if not os.path.exists(path):
    os.makedirs(path)

(width, height) = (130, 100)
count = 1
text = "Person detected"

cam = cv2.VideoCapture(0)

while count < 51:
    count2 = count
    img = cam.read()[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img,(x, y, x+w, y+h), (0, 255, 0), 2)
        Face = gray[y:y+h, x:x+w]
        resImg = cv2.resize(Face, (width, height))
        cv2.imwrite("%s/%s.jpg"%(path, count), resImg)
        count += 1
        cv2.putText(img, text, (10, 20), cv2.FONT_ITALIC, 0.7, (0, 255, 236), 2)
    if count == count2:
        cv2.putText(img , "No "+text, (10, 20), cv2.FONT_ITALIC, 0.7, (0, 255, 236), 2)
    cv2.imshow("Face is detected ", img)
    key = cv2.waitKey(10)
    if key == 27:
        break
print(" Images captured successfully.")
cam.release()