import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import distance


I = cv2.imread('trybik.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = ~I

# thresh = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)
# thresh = thresh[1]
ret, thresh = cv2.threshold(I, 100, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
M = cv2.moments(thresh, 1)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print(cx, cy)  #punkt referencyjny


color = (123, 234, 127)
image = np.zeros((I.shape[0], I.shape[1], 1), np.uint8)
zeros = np.zeros(shape=(I.shape[0], I.shape[1]), dtype=float)
cv2.drawContours(zeros, contours, -1, color)
# contour = cv2.absdiff(I, I_prev)
plt.figure()
plt.gray()
plt.imshow(zeros)
plt.title('Kontur')

sobelx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)

grad = np.sqrt(sobelx**2 + sobely**2)
grad = grad/np.amax(grad)
orient = np.arctan2(sobely, sobelx)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.gray()
plt.imshow(orient)
plt.title('Orientacja gradientu')
plt.subplot(1, 2, 1)
plt.gray()
plt.imshow(grad)
plt.title('Wartość gradientu')

#na gradient sklada sie (grad i orient)


Rtable = [[] for i in range(360)]
for xys in contours:
    for xy in xys:
        #print(xy)
        dist = distance.euclidean([cx, cy], xy[0])
        #print('Uzyskany kąt: ', np.rad2deg(np.arctan2(cy - xy[0, 1], cx - xy[0, 0])))
        #print('Uzyskany orient: ', orient[xy[0, 0]][xy[0, 1]])
        angle = int(np.rad2deg(np.arctan2(cy - xy[0, 1], cx - xy[0, 0])))
        Rtable[int(np.rad2deg(orient[xy[0, 0]][xy[0, 1]]))].append((dist, angle))


I_2 = cv2.imread('trybiki2.jpg')
I_2 = cv2.cvtColor(I_2, cv2.COLOR_BGR2GRAY)
I_2 = ~I_2

# ret, thresh = cv2.threshold(I_2, 100, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
# ####    WYŚWIETLANIE KONTURU CAŁOŚCI ####
# color = (123, 234, 127)
# image = np.zeros((I_2.shape[0], I_2.shape[1], 1), np.uint8)
# zeros = np.zeros(shape=(I_2.shape[0], I_2.shape[1]), dtype=float)
# cv2.drawContours(zeros, contours, -1, color)
# # contour = cv2.absdiff(I, I_prev)


sobelx = cv2.Sobel(I_2, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(I_2, cv2.CV_64F, 0, 1, ksize=5)

grad = np.sqrt(sobelx**2 + sobely**2)
grad = grad/np.amax(grad)
orient = np.arctan2(sobely, sobelx)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 2)
plt.gray()
plt.imshow(orient)
plt.title('Orientacja gradientu całości')
plt.subplot(1, 2, 1)
plt.gray()
plt.imshow(grad)
plt.title('Wartość gradientu całości')

accumulator = np.zeros(I_2.shape) #inicjalizacja tablicy akumulatorów

# for xys in contours:
#     for xy in xys:
#         if grad[xy[0, 0], xy[0, 1]] > 0.5:
#             i = 0
#             for one in Rtable[int(np.rad2deg(orient[xy[0, 0]][xy[0, 1]]))]:
#                 r = one[0]
#                 fi = one[1]
#                 x_c = xy[0, 0] - r * np.cos(np.deg2rad(fi))
#                 y_c = xy[0, 1] - r * np.sin(np.deg2rad(fi))
#                 if x_c < accumulator.shape[0] and y_c < accumulator.shape[1]:
#                     accumulator[int(x_c)][int(y_c)] += 1
#                 i += 1

for x in range(grad.shape[0]):
    for y in range(grad.shape[0]):
        if grad[x, y] > 0.5:
            for one in Rtable[int(np.rad2deg(orient[x][y]))]:
                r = one[0]
                fi = one[1]
                x_c = x + r * np.cos(np.deg2rad(fi))
                y_c = y + r * np.sin(np.deg2rad(fi))
                if x_c < accumulator.shape[0] and y_c < accumulator.shape[1]:
                    accumulator[int(x_c)][int(y_c)] += 1

max_hough = np.where(accumulator.max() == accumulator)
print(max_hough)

plt.figure()
plt.gray()
plt.imshow(accumulator*255/accumulator.max())
plt.title('Przestrzeń Hougha')

plt.figure()
#plt.gray()
plt.imshow(~I_2)
plt.plot(max_hough[1], max_hough[0], '*m')
plt.title('Efekt końcowy')

plt.show()


