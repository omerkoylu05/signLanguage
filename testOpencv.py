import cv2
print(cv2.__version__)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    print("open")
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()