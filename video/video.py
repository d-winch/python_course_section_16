import cv2, time

# Output video to the screen. Press q to exit.

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
