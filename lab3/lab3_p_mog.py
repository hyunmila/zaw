import numpy as np
import cv2

# MOG = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
KNN = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=True)

f = open("lab2/pedestrian/temporalROI.txt", "r")
line = f.readline()
roi_start, roi_end = line.split()
roi_start = int(roi_start)
roi_end = int(roi_end)
step=1
N=60
iN=0

TP = 0
TN = 0
FP = 0
FN = 0

I_prev_read = cv2.imread("lab2/pedestrian/input/in%06d.jpg"%roi_start)
I_next_read = cv2.imread("lab2/pedestrian/input/in%06d.jpg"%(roi_start+1))
# print(roi_start)

# BUF = np.zeros((I_prev_read.shape[0], I_prev_read.shape[1], N), np.uint8)
# print(I_prev_read.shape[0], I_prev_read.shape[1])
# I_prev_g = cv2.cvtColor(I_prev_read, cv2.COLOR_BGR2GRAY)
# I_prev = I_prev_g
# B_diff = cv2.absdiff(I_prev_g, I_prev_g)
# (B_th, B_cons) = cv2.threshold(B_diff, 20, 255, cv2.THRESH_BINARY)
mask = cv2.absdiff(I_prev_read, I_next_read)



for i in range(roi_start+1, roi_end, step):
    I = cv2.imread("lab2/pedestrian/input/in%06d.jpg"%i)
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    # IG = np.float64(IG)
    # lib:
    # BGN = np.where((I_prev<IG), I_prev+1, np.where((I_prev>IG), I_prev-1, I_prev))
    # cons:
    # BGN = np.where(B_cons==1, np.where((I_prev==IG), I_prev, np.where((I_prev<IG), I_prev+1, I_prev-1)), I_prev)
    BS = KNN.apply(IG, mask, -1)
    # BS = MOG.apply(IG, mask, -1)
    # I_diff = cv2.absdiff(IG, BGN)
    cv2.imshow("Diff", BS)

    (T, thresh) = cv2.threshold(BS, 20, 255, cv2.THRESH_BINARY)

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
    # cv2.imshow("Binary3", B3)
    # B4 = cv2.medianBlur(B3, 7)
    cv2.imshow("Binary4", B4)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(B4)
    cv2.waitKey(10)

    # I_prev = BGN
    I_VIS = I # copy of the input image

    if (stats.shape[0]>1):
        tab = stats[1:,4]
        pi = np.argmax(tab)
        pi = pi+1

        cv2.rectangle(I_VIS, (stats[pi,0], stats[pi,1]), (stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]), (255,0,0), 2)
        cv2.putText(I_VIS, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        cv2.putText(I_VIS, "%d" % pi, (int(centroids[pi, 0]), int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        # cv2.imshow("I_VIS", I_VIS)

    G = cv2.imread("lab2/pedestrian/groundtruth/gt%06d.png"%i)
    GT = cv2.cvtColor(G, cv2.COLOR_BGR2GRAY)
    (T, thresh_GT) = cv2.threshold(GT, 160, 255, cv2.THRESH_BINARY)
    # binaryzacja
    # cv2.imshow("G", thresh_GT)
    TP_M = np.logical_and((B4 == 255), (thresh_GT == 255))
    TP_S = np.sum(TP_M)
    TP = TP + TP_S

    FP_M = np.logical_and((B4 == 255), (thresh_GT == 0))
    FP_S = np.sum(FP_M)
    FP = FP + FP_S

    FN_M = np.logical_and((B4 == 0), (thresh_GT == 255))
    FN_S = np.sum(FN_M)
    FN = FN + FN_S

P = TP/(TP+FP) # presission
R = TP/(TP+FN) # recall
F1 = 2*P*R/(P+R)

print("Precission: ", P, "Recall: ", R, "F1: ", F1)
# Median z detekcją cienii: Precission:  0.7248239275826347 Recall:  0.895513745951526 F1:  0.801178422432253
# Median bez detekcji cienii: Precission:  0.7248239275826347 Recall:  0.895513745951526 F1:  0.801178422432253
# KNN: Precission:  0.6875732737194443 Recall:  0.8035776807297473 F1:  0.7410631834136775