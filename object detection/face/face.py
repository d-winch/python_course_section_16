import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("IMG_1485260080402-01-01.jpeg")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grey_img,
scaleFactor = 1.05,
minNeighbors = 8)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4) # img, (x,y), (x,y), BGR, width

cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
