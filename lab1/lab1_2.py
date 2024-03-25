import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# zad 1.2
I = plt.imread('mandril.jpg')
#plt.figure(1)
fig, ax = plt.subplots(1)
plt.imshow(I)
plt.title('Mandril')
plt.axis('off')
# plt.show()

rect = Rectangle((50, 50), 50, 100, fill=False, ec='r');
ax.add_patch(rect)

x = [ 100, 150, 200, 250]
y = [ 50, 100, 150, 200]
plt.plot(x, y, 'r.', markersize=10)
plt.show()


def rgb2gray(I):
    return 0.229*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]

plt.figure(2)
plt.imshow(rgb2gray(I))
plt.title('Mandril_G')
plt.axis('off')
plt.gray()
plt.show()

I_HSV = matplotlib.colors.rgb_to_hsv(I)
plt.figure(3)
plt.imshow(I_HSV)
plt.title('Mandril_G')
plt.axis('off')
plt.show()
