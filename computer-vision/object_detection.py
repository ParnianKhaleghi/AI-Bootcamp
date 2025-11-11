# Convert to grayscale and threshold
import cv2 as cv

img = cv.imread("room.png")
print(img.shape)  # (height, width, channels)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)

# Find contours
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Draw contours
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 1000:  # filter small noise
        cv.drawContours(img, [cnt], -1, (0, 255, 0), 2)

cv.imshow("Detected Objects", img)
cv.waitKey(0)
