import cv2

cascadeClassifierPath = 'haarcascade_frontalface_alt.xml'  # Chemin du Classifier
cascadeClassifier = cv2.CascadeClassifier(cascadeClassifierPath)
cap = cv2.VideoCapture("E:\\masterwisd\\xml\\vid.mp4")  # On récupère la vidéo

while (cap.isOpened()):
    _, frame = cap.read()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion N/B
    detectedFaces = cascadeClassifier.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=10,
                                                       minSize=(20, 20))  # Détection

    for (x, y, width, height) in detectedFaces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Dessin d'un rectangle

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
