import cv2, numpy, os

haar = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar)
cam = cv2.VideoCapture(0)
count = 0

datasets = 'dataset'

(images, labels, id) = ([], [], 0)

x =list(os.walk(datasets))
y=x[0][1]
names = y

for sub in y:
        subjectpath = os.path.join(datasets, sub)
        for filename in os.listdir(subjectpath):
            path = subjectpath+"\\"+filename
            label = id
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)
            labels.append(int(label))
        id += 1


(images, labels) = [numpy.array(lis) for lis in [images,labels]]
(width, height) = (130, 100)

print(images, labels)

model1 = cv2.face.LBPHFaceRecognizer_create()
model2 = cv2.face.FisherFaceRecognizer_create()


model1.train(images, labels)



while True :
    img = cam.read()[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(img,(x, y, x+w, y+h), (0, 255, 0), 2)
        Face = gray[y:y+h, x:x+w]
        resImg = cv2.resize(Face, (width, height))
        prediction = model1.predict(resImg)
        cv2.rectangle(img, (x, y, x + w, y + h), (0, 255, 0), 3)
        if prediction[1] < 800:
            cv2.putText(img, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg", img)
                cnt = 0
        cv2.imshow('FaceRecognition', img)
        key = cv2.waitKey(100)
        if key == 27:
            break
cam.release()
