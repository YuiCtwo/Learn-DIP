import cv2
from admf import adapt_median_filter

trans_img = cv2.imread("boardWithNoise.jpg", cv2.IMREAD_GRAYSCALE)
times = 3
while times > 0:
    trans_img = adapt_median_filter(trans_img, 3, 7)
    times -= 1
    print("Reserved {}-Round".format(times))
print("Finished")
cv2.imshow("自适应滤波", trans_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
