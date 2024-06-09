import cv2
import numpy as np


video_path = 'video_2.mp4'
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # color range for yellow and red balls in HSV
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    # output video
    output_path = 'annotated_video_2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # process each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # create masks for yellow and red colors
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # find contours for yellow balls
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_yellow:
            if cv2.contourArea(contour) > 50:  # filter out small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, 'yellow ball', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # find contours for red balls
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_red:
            if cv2.contourArea(contour) > 50:  # filter out small areas
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if area < 500:  # smaller red ball
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'red ball small', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:  # larger red ball
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 128), 2)
                    cv2.putText(frame, 'red ball large', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 2)
        
        out.write(frame)


    cap.release()
    out.release()

