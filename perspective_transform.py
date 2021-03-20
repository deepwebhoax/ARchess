import cv2
import numpy as np
import matplotlib.pyplot as plt
# from chessboard_detection import inference

# inference('input/31.jpg')
img = cv2.imread('input/31.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[97,241],[872,207],[110,1025],[860,1030]])
pts2 = np.float32([[0,0],[500,0],[0,500],[500,500]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(500,500))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()