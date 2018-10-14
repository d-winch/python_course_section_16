import cv2

# Press q to exit.

first_frame = None

video = cv2.VideoCapture(0)

while True:

    # Get frame, convert to greyscale, apply blur
    check, frame = video.read()
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_img = cv2.GaussianBlur(grey_img, (21, 21), 0) # img, (w, h), standard deviation

    # Set first frame if not assigned, restart loop
    if first_frame is None:
        first_frame = grey_img
        continue

    # Get difference from first frame and current, apply threshhold and dilate to smooth noise
    delta_frame = cv2.absdiff(first_frame, grey_img)
    thresh_delta_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] # img, threshold value, assigned value, method
    thresh_delta_frame = cv2.dilate(thresh_delta_frame, None, iterations = 2) # img, kernel, iterations

    (_, cnts, _) = cv2.findContours(
        thresh_delta_frame.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    ) # img, mode, method
    # RETR_EXTERNAL = retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE = compresses horizontal, vertical, and diagonal segments and leaves only their end points

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) #img, (x,y), (x,y), (BGR), width

    cv2.imshow("Capture", frame)
    cv2.imshow("Diff", delta_frame)
    cv2.imshow("Diff", thresh_delta_frame)

    key = cv2.waitKey(1)
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
