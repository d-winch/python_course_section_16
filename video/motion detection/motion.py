import cv2, pandas
from datetime import datetime

# Press q to exit.

first_frame = None
status_list = [None, None] # Avoid index error
time_list = []
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

while True:

    # Motion status
    status = 0
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
    thresh_delta_frame = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1] # img, threshold value, assigned value, method
    thresh_delta_frame = cv2.dilate(thresh_delta_frame, None, iterations = 2) # img, kernel, iterations

    # Find the contours in the image
    (_, cnts, _) = cv2.findContours(
        thresh_delta_frame.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    ) # img, mode, method
    # RETR_EXTERNAL = retrieves only the extreme outer contours
    # CHAIN_APPROX_SIMPLE = compresses horizontal, vertical, and diagonal segments and leaves only their end points

    # if contour area > 5000, draw green bounding rectangle
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) # img, (x,y), (x,y), (BGR), width

    # Append status to status_list
    # If status has changed since last frame, add timestamp to list
    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        time_list.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        time_list.append(datetime.now())

    # Display frames
    cv2.imshow("Capture", frame)
    cv2.imshow("Diff", delta_frame)
    cv2.imshow("Threshold", thresh_delta_frame)

    # Display until q is pressed
    key = cv2.waitKey(1)
    if key==ord('q'):
        # If motion detected when quitting, add timestamp of closing
        if status == 1:
            time_list.append(datetime.now())
        break

# For all times in list, step 2
for i in range(0, len(time_list), 2):
    # Ignore index - do not use the index labels
    df = df.append({"Start":time_list[i], "End":time_list[i+1]}, ignore_index = True)

# Write to file
df.to_csv("Timestamps.csv")

# Release the camera and close windows
video.release()
cv2.destroyAllWindows()
