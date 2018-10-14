import cv2, time

# Output video with rectangle around detected faces. Press q to exit.

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey_img,
    scaleFactor = 1.05,
    minNeighbors = 10)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4) # img, (x,y), (x,y), BGR, width
    return img

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    faces = detect_faces(frame)
    cv2.imshow("Capture", faces)
    key = cv2.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
