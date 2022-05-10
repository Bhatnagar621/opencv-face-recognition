import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

c = input('Do you wish to train a new image though web cam (y/n): ')

if c=='y' or c=='Y':
  cap = cv2.VideoCapture(0)
  label = input('Enter a name for the person: ')
  while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if label in label_ids:
      pass
    else:
      label_ids[label] = current_id
      current_id+=1
    id_ = label_ids[label]
    cv2.imwrite('{}.jpg'.format(label), gray)
    pil_image = cv2.imread('{}.jpg'.format(label)) #grayscale the image
    image_array = np.array(pil_image, 'uint8')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
      roi_gray = gray[y:y+h, x:x+w]
      x_train.append(roi_gray)
      y_labels.append(id_)
      color = (255, 0, 0)
      stroke = 2
      end_cord_y = y+h
      end_cord_x = x+w
      cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke) 
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()    

for root, dirs, files in os.walk(image_dir):
  for file in files:
    if file.endswith('png') or file.endswith('jpg') or file.endswith('webp'):
      path = os.path.join(root, file)
      label = os.path.basename(root).replace(' ', '-').lower()
      #print(label, path)
      if label in label_ids:
        pass
      else:
        label_ids[label] = current_id
        current_id+=1

      id_ = label_ids[label]
      #print(label_ids)
      #x_train.append(path)      #verify image, turn into numpy array, GRAY
      #y_labels.append(label)    #some number for a label
      pil_image = Image.open(path).convert('L')#grayscale the image
      image_array = np.array(pil_image, 'uint8')
      #print(image_array)
      faces = face_cascade.detectMultiScale(image_array)

      for (x, y, w, h) in faces:
        roi = image_array[y:y+h, x:x+w]
        x_train.append(roi)
        y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open('label.pickle', 'wb') as f:
  pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainer.yml')