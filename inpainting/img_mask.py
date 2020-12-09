import cv2
import numpy as np

org = cv2.cvtColor(cv2.imread('examples/female.png'), cv2.COLOR_BGR2RGB)
im = org.copy()
mask = np.zeros_like(im)
mask[500:700, 600:800, :] = 255 ## height, width, channel
im[500:700, 600:800, :] = 255

#plt.imshow(im)
cv2.imwrite('examples/pic3.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
cv2.imwrite('examples/mask3.jpg', mask)