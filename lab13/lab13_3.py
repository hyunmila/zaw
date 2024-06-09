import cv2
import numpy as np


events_file_path = 'events.txt'
events = []

with open(events_file_path, 'r') as file:
    while True:
        line = file.readline()
        if not line:
            break
        timestamp, x, y, polarity = line.split()
        timestamp = float(timestamp)
        x = int(x)
        y = int(y)
        polarity = int(polarity) if int(polarity) > 0 else -1
        if 1.0 < timestamp < 2.0:
            events.append([timestamp, x, y, polarity])


def event_frame(x_coords, y_coords, polarities, image_shape):
    image = np.ones(image_shape) * 127
    image = image.astype(np.uint8)

    for x, y, polarity in zip(x_coords, y_coords, polarities):
        image[y, x] = 255 if polarity > 0 else 0

    return image

# duration of data aggregation
tau = 0.01  # 10 ms

temp_timestamps = []
temp_x_coords = []
temp_y_coords = []
temp_polarities = []

for event in events:
    timestamp, x, y, polarity = event
    temp_timestamps.append(timestamp)
    temp_x_coords.append(x)
    temp_y_coords.append(y)
    temp_polarities.append(polarity)

    if temp_timestamps[-1] - temp_timestamps[0] >= tau:
        image = event_frame(temp_x_coords, temp_y_coords, temp_polarities, (180, 240))
        cv2.imshow('Event Frame', image)
        cv2.waitKey(10)

        temp_timestamps = []
        temp_x_coords = []
        temp_y_coords = []
        temp_polarities = []

cv2.destroyAllWindows()

"""
Wartości tau wpływają na liczbę zdarzeń uwzględnionych w każdej 
ramce zdarzeń
    - tau = 1ms, ramki będą bardziej szczegółowe, pokazując 
    mniejsze fragmenty ruchu
    - tau = 10ms, ramki będą bardziej płynne, pokazując większe 
    fragmenty ruchu
    - tau = 100ms, ramki będą jeszcze bardziej płynne, ale mogą 
    utracić szczegóły szybkich ruchów.
"""