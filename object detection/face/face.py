import sys
import cv2

print(sys.argv)

if(len(sys.argv) < 4):
    print(
        "\nPlease provide arguments - filepath, scale factor, min neighbors\n"
        "Suggested - scale factor 1.05, min neighbors 5 to 20\n"
        "E.g., python .\\face.py .\\business.jpeg 1.05 15\n\n"
        "SCRIPT EXPECTS CORRECT INPUT TYPES AT CURRENT\n"
    )
    exit(0)

filePath, scaleFactor, minNeighbors = sys.argv[1:]

print(filePath)
print(scaleFactor)
print(minNeighbors)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread(filePath)
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grey_img,
scaleFactor = float(scaleFactor),
minNeighbors = int(minNeighbors))

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4) # img, (x,y), (x,y), BGR, width

img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)) )
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
