import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

events_file_path = 'events.txt'

# read the events.txt file and parse the data
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
        polarity = int(polarity)
        if timestamp < 1.0:
            events.append([timestamp, x, y, polarity])
        else:
            break

# split the events into separate lists for timestamps, x, y, and polarity
timestamps = [event[0] for event in events]
x_coords = [event[1] for event in events]
y_coords = [event[2] for event in events]
polarities = [event[3] for event in events]

# analyze the events
num_events = len(events)
first_timestamp = timestamps[0]
last_timestamp = timestamps[-1]
max_x = max(x_coords)
min_x = min(x_coords)
max_y = max(y_coords)
min_y = min(y_coords)
positive_polarity_count = sum(1 for p in polarities if p > 0)
negative_polarity_count = num_events - positive_polarity_count

print(f"Number of events: {num_events}")
print(f"First timestamp: {first_timestamp}")
print(f"Last timestamp: {last_timestamp}")
print(f"Max x: {max_x}, Min x: {min_x}")
print(f"Max y: {max_y}, Min y: {min_y}")
print(f"Positive polarity events: {positive_polarity_count}")
print(f"Negative polarity events: {negative_polarity_count}")

# separate events based on polarity for visualization
positive_events = [event for event in events if event[3] > 0]
negative_events = [event for event in events if event[3] <= 0]

pos_timestamps = [event[0] for event in positive_events]
pos_x_coords = [event[1] for event in positive_events]
pos_y_coords = [event[2] for event in positive_events]

neg_timestamps = [event[0] for event in negative_events]
neg_x_coords = [event[1] for event in negative_events]
neg_y_coords = [event[2] for event in negative_events]

# visualize the event data with a 3D chart
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pos_x_coords, pos_y_coords, pos_timestamps, c='r', marker='o', label='Positive Polarity')
ax.scatter(neg_x_coords, neg_y_coords, neg_timestamps, c='b', marker='x', label='Negative Polarity')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Timestamp')
ax.legend()

plt.show()

# 2.1 plot 3D chart for the first 8000 events
first_8000_events = events[:8000]
first_8000_pos_events = [event for event in first_8000_events if event[3] > 0]
first_8000_neg_events = [event for event in first_8000_events if event[3] <= 0]

first_8000_pos_timestamps = [event[0] for event in first_8000_pos_events]
first_8000_pos_x_coords = [event[1] for event in first_8000_pos_events]
first_8000_pos_y_coords = [event[2] for event in first_8000_pos_events]

first_8000_neg_timestamps = [event[0] for event in first_8000_neg_events]
first_8000_neg_x_coords = [event[1] for event in first_8000_neg_events]
first_8000_neg_y_coords = [event[2] for event in first_8000_neg_events]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(first_8000_pos_x_coords, first_8000_pos_y_coords, first_8000_pos_timestamps, c='r', marker='o', label='Positive Polarity')
ax.scatter(first_8000_neg_x_coords, first_8000_neg_y_coords, first_8000_neg_timestamps, c='b', marker='x', label='Negative Polarity')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Timestamp')
ax.legend()

# rotate the chart
ax.view_init(elev=30, azim=60)  
plt.show()

# 2.2 plot 3D chart for events with timestamp between 0.5 and 1
filtered_events = [event for event in events if 0.5 <= event[0] < 1]
filtered_pos_events = [event for event in filtered_events if event[3] > 0]
filtered_neg_events = [event for event in filtered_events if event[3] <= 0]

filtered_pos_timestamps = [event[0] for event in filtered_pos_events]
filtered_pos_x_coords = [event[1] for event in filtered_pos_events]
filtered_pos_y_coords = [event[2] for event in filtered_pos_events]

filtered_neg_timestamps = [event[0] for event in filtered_neg_events]
filtered_neg_x_coords = [event[1] for event in filtered_neg_events]
filtered_neg_y_coords = [event[2] for event in filtered_neg_events]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(filtered_pos_x_coords, filtered_pos_y_coords, filtered_pos_timestamps, c='r', marker='o', label='Positive Polarity')
ax.scatter(filtered_neg_x_coords, filtered_neg_y_coords, filtered_neg_timestamps, c='b', marker='x', label='Negative Polarity')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Timestamp')
ax.legend()

# rotate the chart
ax.view_init(elev=30, azim=60) 
plt.show()

"""
1. Ciąg trwa 1 sekundę
2. Rozdzielczość znaczników czasu zdarzeń jest w mikrosekundach, co wskazuje na 
    precyzję zmiennoprzecinkową
3. Różnica czasowa między kolejnymi zdarzeniami zależy od szybkości, z jaką zmiany 
    są wykrywane przez kamerę zdarzeń, na co ma wpływ prędkość ruchu i czułość kamery
4. Dodatnia polaryzacja zdarzeń wskazuje na wzrost jasności (zdarzenia "światło 
    włączone"), podczas gdy ujemna polaryzacja zdarzeń wskazuje na spadek jasności 
    (zdarzenia "światło wyłączone")
5. Kierunek ruchu obiektów można wywnioskować, obserwując trajektorie zdarzeń w 
    czasie na wykresie 3D dla zakresu znaczników czasu między 0.5 a 1 sekundą

"""
