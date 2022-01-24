import cv2
import pyttsx3


def Speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  
    engine.say(text) 
    engine.runAndWait()

Speak("hello")


mask_cascade = cv2.CascadeClassifier('mask.xml')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
frame_count=0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = mask_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0: cv2.putText(frame,"GET IN THE FRAME",(40,40),4,frame.shape[0]  / frame.shape[1],(255,255,255))
    else: frame_count+=1
    if (frame_count==100):
        Speak("mask found check complete")
        break
    for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
    cv2.imshow('Mask Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()