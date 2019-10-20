 # -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:44:46 2018

@author: bunte
"""

import numpy as np
import cv2
import mysql.connector
face_cascade = cv2.CascadeClassifier('C:\\Users\\bunte\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("G:\\Yash_CS\\Inceptra\\Training Data.yml")

#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontColor = (255, 255, 255)
# Open database connection
db = mysql.connector.connect(host='localhost',database='bank',user='root',password='********' )        
# prepare a cursor object using cursor() method
cursor = db.cursor() 
while 1:
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        
        if id==1:
            id="Yash"
            amt=input("enter the amount")
            cursor.execute("update reg set balance =balance-%s where regid='1'"%(amt))
            db.commit()
            break
        if id==2:
            id="Sai Raja"
            amt=input("enter the amount")
            cursor.execute("update reg set balance =balance-%s where regid='2'"%(amt))
            db.commit()
            break
        #if id==3:
           # id="Navyya"
            #amt=input("enter the amount")
            #cursor.execute("update reg set balance =balance-%s where regid='3'"%(amt))
            #db.commit()
            #break
        #if id==4:
            #id="Pooja"
            #amt=input("enter the amount")
            #cursor.execute("update reg set balance =balance-%s where regid='4'"%(amt))
            #db.commit()
            #break
            
        
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
        cv2.putText(img, str(id), (x,y+h), fontface,fontscale,fontColor)
    
    
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()

