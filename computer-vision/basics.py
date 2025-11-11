import cv2

img = cv2.imread("dog.jpg")

# Show image
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize
resized = cv2.resize(img, (300, 300))

# Crop
cropped = img[50:200, 100:300]

cv2.imshow("Gray", gray)
cv2.waitKey(0)
cv2.imshow("Cropped", cropped)
cv2.waitKey(0)
# cv2.destroyAllWindows()
