import cv2 as cv

img = cv.imread("room.png")
print(img.shape)  # (height, width, channels)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)
cv.imshow("Edges", edges)
cv.waitKey(0)
