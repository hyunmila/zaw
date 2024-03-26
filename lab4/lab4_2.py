import cv2
import numpy as np
import matplotlib.pyplot as plt




def of(I, J, u0, v0, W2=1, dY=1, dX=1):
    I = I[-1]
    J = J[-1]
    # YY, XX = I.shape[:2]  # height, width
    Y = I.shape[0]
    X = I.shape[1]
    u1 = np.zeros((Y, X))
    v1 = np.zeros((Y, X))
    for j in range(W2+1, Y-W2-1):
        for i in range(W2+1, Y-W2-1):
            IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
            min_dist = 10000000
            for jj in range(j - dY, j + dY + 1):
                for ii in range(i - dX, i + dX + 1):
                    #if j_1 < (J.shape[0] - W2) and i_1 < (J.shape[1] - W2) and i_1 > W2 and
                    JO = np.float32(J[jj + int(u0[j, i]) - W2:jj + int(u0[j, i]) + W2 + 1, ii + int(v0[j, i]) - W2:ii + int(v0[j, i]) + W2 + 1])
                    l = np.sum(np.sqrt((np.square(JO - IO))))
                    if (l < min_dist):
                        min_dist = l
                        u1[j, i] = jj + u0[j, i] - j
                        v1[j, i] = jj + v0[j, i] - i
    return u1, v1

def vis_flow(u, v, X, Y, name):
    magnitude, angle = cv2.cartToPolar(u,v)
    I_HSV = np.zeros((Y, X, 3), np.uint8)
    I_HSV[:,:,0] = angle*90/np.pi
    I_HSV[:,:,1] = 255
    I_HSV[:,:,2] = cv2.normalize(magnitude, 0, 255)
    cv2.imshow(str(name), I_HSV)
    # I_OUT = cv2.cvtColor(I_HSV, cv2.COLOR_HSV2RGB)
    # cv2.imwrite("out.png", I_OUT)
    # cv2.imshow("out", I_OUT)
    cv2.waitKey(10)

def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0,0), fx=0.5, fy=0.5))
    return images

I = cv2.imread('lab4\I.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
J = cv2.imread('lab4\J.jpg')
J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)

# cv2.imshow("I.jpg", I)
# cv2.imshow("J.jpg", J)

# Diff = cv2.absdiff(I, J)
# cv2.imshow("Difference", Diff)

# cv2.waitKey(10)

# YY, XX = I.shape[:2]  # height, width
# u = np.zeros((YY, XX))
# v = np.zeros((YY, XX))

K = 3 #ilosc skalowan

IP = pyramid(I, K)
JP = pyramid(J, K)

u0 = np.zeros(IP[-1].shape, np.float32) #wektor[-1] <- dostep do ostatniego elementu
v0 = np.zeros(JP[-1].shape, np.float32)

u1, v1 = of(IP, JP, u0, v0)
for k in range(1, K):
    v1 = cv2.resize(v1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    u1 = cv2.resize(u1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    #u1, v1 = of(IP, JP, u1, v1)
    k = k + 1

