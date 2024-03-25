import cv2
import numpy as np
import matplotlib.pyplot as plt

# zad1.1
I = cv2.imread('mandril.jpg')
# cv2.imshow("Mandril", I)
# cv2.waitKey(1000)
# cv2.imwrite("m.png", I)
# print(I.shape)
# print(I.size)
# print(I.dtype)

# zad 1.3
IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# cv2.imshow("Mandril_IG", IG)
# cv2.waitKey(1000)
# cv2.imshow("Mandril_IHSV", IHSV)
# cv2.waitKey(1000)
IH = IHSV[:,:,0]
IS = IHSV[:,:,1]
IV = IHSV[:,:,2]

# print(IH)
# print(IS)
# print(IV)

# zad 1.4
height, width = I.shape[:2]
scale = 1.75
Ix2 = cv2.resize(I,(int(scale*height), int(scale*width)))
# cv2.imshow("Big_Mandril", Ix2)
# cv2.waitKey(1000)

# zad 1.5
J = cv2.imread('lena.png')
JG = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)

gray_add = JG + IG
# cv2.imshow("Add", gray_add)
# cv2.waitKey(1000)
cv2.imwrite("add.png", gray_add)

gray_sub = IG - JG
# cv2.imshow("Sub", gray_sub)
# cv2.waitKey(1000)
cv2.imwrite("sub.png", gray_sub)

gray_mul = IG.astype(np.uint16) * JG.astype(np.uint16)
# cv2.imshow("Mul", gray_mul)
# cv2.waitKey(1000)
cv2.imwrite("mul.png", gray_mul)

C = 0.5 * JG + 0.75 * IG
# cv2.imshow("C",np.uint8(C))
# cv2.waitKey(1000)
cv2.imwrite("c.png", np.uint8(C))

gray_mod = cv2.absdiff(JG, IG)
# cv2.imshow("Mod", gray_mod)
# cv2.waitKey(1000)
cv2.imwrite("mod.png", gray_mod)

gray_mod_2 = abs(JG - IG)
# gray_mod_2 = abs(JG - IG)
cv2.imwrite("mod2.png", gray_mod_2)
# cv2.imshow("Mod_2", gray_mod_2)
# cv2.waitKey(100000)
# cv2.destroyAllWindows()


# # zad 1.6
def hist(img): # metoda 1
    h = np.zeros((256, 1), np.float32)
    height, width = img.shape[:2]
    for x in range(width):
        for y in range(height):
            h[img[x, y]] += 1

    return h

plt.figure()
plt.subplot(1,2,1)
plt.plot(hist(IG))
plt.title('Histogram 1')
plt.axis('on')

hist2 = cv2.calcHist([IG], [0], None, [256], [0, 256]) # metoda 2

plt.subplot(1,2,2)
plt.plot(hist2)
plt.title('Histogram 2')
plt.axis('on')
plt.show()

plt.figure()
plt.subplot(1,2,1)
IGE = cv2.equalizeHist(IG)
plt.hist(IGE)
plt.title('Wyrownanie 1')
plt.axis('on')

plt.subplot(1,2,2)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
I_CLAHE = clahe.apply(IG)
plt.hist(I_CLAHE)
plt.title('Wyrownanie 2')
plt.axis('on')
plt.show()
