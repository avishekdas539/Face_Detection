import cv2
import os
import uuid



IMAGES_PATH = "Data\\images"
# IMAGES_PATH = "Data\\train\\images"
# IMAGES_PATH = "Data\\val\\images"
# IMAGES_PATH = "Data\\test\\images"
number_of_images = 50

cap = cv2.VideoCapture(0)
for i in range(number_of_images):
    print(f"Capturing Image Number {i}")
    ret, frame = cap.read()
    img_name = os.path.join(IMAGES_PATH, f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(img_name, frame)
    cv2.imshow("Frame", frame)
    cv2.waitKey(500)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()