import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('vid1_IR.wm')

while(cap.isOpened()):
    ret, frame = cap.read()
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    B, thresh = cv2.threshold(G, 50, 255, cv2.THRESH_BINARY)

    B0 = cv2.medianBlur(thresh, 7)
    kernel = np.ones((3,3), np.uint8)
    B1 = cv2.erode(B0, kernel, iterations=1)
    # cv2.imshow("Binary1", B1)
    kernel = np.ones((4, 4), np.uint8)
    B2 = cv2.dilate(B1, kernel, iterations=2)
    # cv2.imshow("Binary2", B2)
    kernel = np.ones((2, 2), np.uint8)
    B3 = cv2.erode(B2, kernel, iterations=1)
    kernel = np.ones((3,3), np.uint8)
    B4 = cv2.morphologyEx(B3, cv2.MORPH_CLOSE, kernel, iterations=1)


    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(B4)
    # stats -- left, right, width, heigth, P

    # cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

    if (stats.shape[0] > 1):  # znalezienie obiektów
        for x in range(0, stats.shape[0]):
            if (stats[x, 3] > 0.5*stats[x, 2]) and (stats[x, 4] > 500): # rozpatrywanie najwiekszych mozliwych obiektow
                
                left = stats[x, 0]
                right = stats[x, 0] + stats[x, 2]
                top = stats[x, 1]
                bottom = stats[x, 1] + stats[x, 3]

                for y in range(0, stats.shape[0]):

                    left_y = stats[y, 0]
                    right_y = stats[y, 0] + stats[y, 2]
                    top_y = stats[y, 1]
                    bottom_y = stats[y, 1] + stats[y, 3]

                    if (stats[y, 3] > stats[y, 2]) and (stats[y, 4] > 10) and (
                        left-5 <= (left_y + (1/2)*stats[y, 2]) <= right+5 and (
                        top_y < top - 10)) or (left <= (left_y + (1/2)*stats[y, 2]) <= right and (bottom_y > bottom + 10)):
                        # dolaczenie pozostalych bloczkow
                            
                        left = min(left, left_y)
                        top = min(top, top_y)
                        right = max(right, right_y)
                        bottom = max(bottom, bottom_y)
                        # P = (bottom - top) * (right - left)
                
                cv2.rectangle(G, (left, top), (right, bottom), (255, 0, 0), 2)

    cv2.imshow("G", G)
    # cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'): # przerwanie petli po wcisnieciu klawisza ’q’
        break

cap.release()
cv2.destroyAllWindows()