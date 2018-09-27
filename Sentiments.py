# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:47:16 2018

@author: Sandesh
"""

import pandas as pd
import numpy as np
import cv2
from functions import *
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
#import dlib
import time
from statistics import mode,median
import sys
import warnings

class sentiment():
  if not sys.warnoptions:
      warnings.simplefilter("ignore")

  def hist(self , image):
      hist,bins = np.histogram(image.flatten(),256,[0,256])
       
      cdf = hist.cumsum()
      cdf_normalized = cdf * hist.max()/ cdf.max()
      
      '''plt.plot(cdf_normalized, color = 'b')
      plt.hist(img.flatten(),256,[0,256], color = 'r')
      plt.xlim([0,256])
      plt.legend(('cdf','histogram'), loc = 'upper left')
      plt.show()'''
      cdf_m = np.ma.masked_equal(cdf,0)
      cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
      cdf = np.ma.filled(cdf_m,0).astype('uint8')
      img2 = cdf[image]
      return img2
  from keras.callbacks import Callback
  from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

  '''
  def train_model():   
   classifier = Sequential()

  # Step 1 - Convolution
    classifier.add(Convolution2D(64, 3, 3, input_shape = (48, 48,3), activation = 'relu'))

  # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
   #classifier.add(Convolution2D(32, 4, 4, activation = 'relu'))

  # Adding a second convolutional layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

  # Step 3 - Flattening
    classifier.add(Flatten())

  # Step 4 - Full connection
    classifier.add(Dense(output_dim = 512, activation = 'relu'))
   #classifier.add(Dense(output_dim=128,activation='relu'))
    classifier.add(Dense(output_dim=28,activation='relu'))
    classifier.add(Dense(output_dim = 7, activation = 'sigmoid'))

  # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy',precision])

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                     shear_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('imagedata/training_set',
                                                   target_size = (48, 48),
                                                   batch_size = 32,
                                                   class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('imagedata/test_set',
                                              target_size = (48, 48),
                                              batch_size = 32,
                                              class_mode = 'categorical')


    classifier.fit_generator(training_set,
                           samples_per_epoch = 28658,
                           nb_epoch = 25,
                           validation_data = test_set,
                           nb_val_samples = 8523)
    classifier.summary()
    return classifier
  '''

  #Prediction
  #class_labels = {v: k for k, v in training_set.class_indices.items()}
  def predict(self , img,classifier):
   img = cv2.resize(img,(48,48))
   img = np.expand_dims(img,axis=0)
   if(np.max(img)>1):
      img = img/255.0
   
   prediction = classifier.predict_classes(img)
   return prediction

   
   #Saving model
  def save_model(self , classifier,name='model_1'):
   model_json = classifier.to_json()
   with open("%s.json"%name, "w") as json_file:
      json_file.write(model_json)
   # serialize weights to HDF5
   classifier.save_weights("%s.h5"%name)
   print("Saved model to disk")

  #Loading model
  def load_jason_model(self , name='model_1'):
   json_file = open('%s.json'%name, 'r')
   loaded_model_json = json_file.read()
   json_file.close()
   loaded_model = model_from_json(loaded_model_json)
   # load weights into new model
   loaded_model.load_weights("%s.h5"%name)
   print("Loaded model from disk")
   loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   return loaded_model

  #detector = dlib.get_frontal_face_detector()
  
   #print(EMOTIONS[mode(arr)])
   



  def softmax(self , a):
      expA=np.exp(a)
      return expA/expA.sum(axis=1,keepdims=True)


      

  def app(self):
      print('Starting app....')
      model=load_jason_model('bigger_model')
      face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
      #int_time=time.time()
      prev_time=time.time()-30
      current_time=time.time()
      print('Before '+str(prev_time))
      cap = cv2.VideoCapture('video1.mp4')
      cap.set(cv2.CAP_PROP_FPS, 5)
      while(True):
         ret,img=cap.read()
         if((int(current_time)-int(prev_time))>=30):
             count=0
             arr=[]
             while(count<9):
              ret,img=cap.read()
              img2=hist(img)
              gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
              faces = face_cascade.detectMultiScale(gray, 1.3, 5)
              for (x,y,w,h) in faces:
                 cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),1)
                 image=img[y:y+h,x:x+w]
                 image=cv2.resize(image,(48,48))
                 image=np.expand_dims(image,axis=0)
                 if(np.max(image)>1):
                   image = image/255.0
                 #prediction = model.predict(image)
                 #prediction=softmax(prediction)
                 c=model.predict_classes(image)
                 arr.append(c[0])
                 font=cv2.FONT_HERSHEY_COMPLEX
                 cv2.putText(img,EMOTIONS[c[0]],(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
                 count+=1
                 #print(count)
              cv2.imshow('img',img)
              k = cv2.waitKey(30) & 0xff
              if k==27:
                  break
             try:
               print(EMOTIONS[mode(arr)])
             except:
                 print(EMOTIONS[median(arr)])
             current_time=time.time()
             prev_time=current_time
             print(prev_time)
         else:
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                break
            current_time=time.time()
         '''if(int(current_time)-int(int_time)>=30):
             break'''
         
      cap.release()
      cv2.destroyAllWindows()
         

     
  def emotion(self , cap , model , EMOTIONS):
      arr=[]
      count=0
      while(count<5):
            ret,img=cap.read()
            img2=self.hist(img)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
              cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),1)
              image=img[y:y+h,x:x+w]
              image=cv2.resize(image,(48,48))
              image=np.expand_dims(image,axis=0)
              if(np.max(image)>1):
               image = image/255.0
              prediction = model.predict(image)
              c=model.predict_classes(image)
              arr.append(c[0])
              font=cv2.FONT_HERSHEY_COMPLEX
              cv2.putText(img,EMOTIONS[c[0]],(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
              count+=1
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                  break
      try:
          return mode(arr)
      except:
          return median(arr)
      



def main(): 
  obj_sentiment = sentiment()
  s=10
  EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']
  model=obj_sentiment.load_jason_model('bigger_model')
  face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
  prev_time=time.time()-s
  current_time=time.time()
  intial_time = time.time()
  cap = cv2.VideoCapture(1)
  #cap.set(cv2.CAP_PROP_FPS, 5)
  while(cap.isOpened()):
      ret,img=cap.read()
      if((int(current_time)-int(prev_time))>=s):
          e=obj_sentiment.emotion(cap,model , EMOTIONS)
          print(EMOTIONS[e])
          current_time=time.time()
          prev_time=current_time 
      else:
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff 
        if k==27  or time.time() - intial_time >= 15:
            break  
        current_time=time.time()
  cap.release()
  cv2.destroyAllWindows()        




if __name__=='__main__':
  main()
