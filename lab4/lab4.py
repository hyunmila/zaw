import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread("I.jpg") # wcześniejsza ramka
# I_in = cv2.imread("I.jpg") # wcześniejsza ramka
J = cv2.imread("J.jpg") # późniejsza ramka

scale = 0.5

I = cv2.resize(I,(int(scale*I.shape[1]), int(scale*I.shape[0])))
I_in = I
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
J = cv2.resize(J,(int(scale*J.shape[1]), int(scale*J.shape[0])))
J = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
Diff = cv2.absdiff(I, J)
# cv2.namedWindow("Diff", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("Diff", Diff)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(I.shape[0], I.shape[1]) # wysokosc x szerokosc

# size = 7
W2 = 5
dX = 5
dY = 5
l=[]
# l_min_list = []
u = np.zeros((I.shape[0], I.shape[1]))
v = np.zeros((I.shape[0], I.shape[1]))

for j in range(W2+1, I.shape[0]-W2-1):
    for i in range(W2+1, I.shape[1]-W2-1):
        I0 = np.float32(I[j-W2:j+W2+1, i-W2:i+W2+1])
        min_dist = 10000000
        for jj in range(j-dY, j+dY+1):
            for ii in range(i-dX, i+dX+1):
                J0 = np.float32(J[jj-W2:jj+W2+1, ii-W2:ii+W2+1])
                # print(jj-W2,jj+W2+1, ii-W2,ii+W2+1)
                # print(i-dX, i+dX+1)
                l = np.sqrt(np.sum((np.square(J0-I0))))
                if (l<min_dist):
                    min_dist=l
                    u[j, i] = jj-j
                    v[j, i] = ii-i
                    
                    
magnitude, angle = cv2.cartToPolar(u,v)

I_HSV = cv2.cvtColor(I_in.astype(np.uint8), cv2.COLOR_RGB2HSV, 3)
I_HSV[:,:,0] = angle*90/np.pi
I_HSV[:,:,1] = 255
I_HSV[:,:,2] = cv2.normalize(magnitude, 0, 255)
cv2.imshow("out", I_HSV)
I_OUT = cv2.cvtColor(I_HSV, cv2.COLOR_HSV2RGB)
cv2.imwrite("out.png", I_OUT)
# cv2.imshow("out", I_OUT)
cv2.waitKey(0)
# plt.quiver(u, v)
# plt.gca().invert_yaxis()
# plt.show()