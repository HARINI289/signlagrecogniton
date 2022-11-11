from keras.models import load_model
import cv2
import numpy as np
classes = ['D',
 '3',
 'T',
 '6',
 'P',
 '1',
 '2',
 '7',
 'G',
 'W',
 '5',
 'L',
 'R',
 'C',
 'H',
 'K',
 'J',
 'B',
 'Y',
 '8',
 'U',
 'F',
 'O',
 'N',
 'Z',
 'E',
 'V',
 'X',
 'A',
 'S',
 '9',
 'I',
 '4',
 'M',
 '0',
 'Q']

def make_pred(img_path, model_path):
  model = load_model(model_path)
  img = cv2.imread(img_path)
  img_resied = cv2.resize(img, (100,100))
  img_reshape = img_resied.reshape(-1, 100, 100, 1)
  res = model.predict(img_reshape)
  print(classes[np.argmax(res)])

make_pred("/content/original_images/7/143.jpg","model1.h5")
