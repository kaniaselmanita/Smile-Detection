import cv2


def detect_smile(smile_cascade, smile_roi):
    smiles = smile_cascade.detectMultiScale(smile_roi, scaleFactor=1.7, minNeighbors=39)
    if len(smiles) > 0:
        return True
    else:
        return False
   

def main(capture):
    face_dataset = 'haarcascade_frontalface_alt.xml'
    smile_dataset = 'haarcascade_smile.xml'
    body_dataset = 'haarcascade_upperbody.xml'

    face_cascade = cv2.CascadeClassifier(face_dataset)
    smile_cascade = cv2.CascadeClassifier(smile_dataset)

    while True:
        ret, frame = capture.read()

        # Upper body detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (fx, fy, fw, fh) in faces:
            # Smile detection within the face region
            smile_roi = gray[fy:fy+fh, fx:fx+fw]
            if detect_smile(smile_cascade, smile_roi):
                cv2.putText(frame, 'FACE & SMILE', (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'FACE & NO SMILE', (fx, fy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = cv2.VideoCapture(2)  
    main(camera)