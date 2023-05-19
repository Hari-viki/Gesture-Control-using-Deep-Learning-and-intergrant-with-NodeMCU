import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import requests
model=load_model('model.h5')

word={0:"0",1:"1"}
print("Hello")

video=cv2.VideoCapture(0)

while 1:
    _,frame=video.read()
    # frame = cv2.imread(str(file.name))
    reta=cv2.resize(frame,(500,300))
    cv2.imwrite('1.jpg',reta)
    from tensorflow.keras.preprocessing import image
    img=image.load_img('1.jpg',target_size=(70,70))
    resize=image.img_to_array(img)
    test_image=np.expand_dims(resize , axis = 0)     
    predict=model.predict( test_image )
    # print(predict)
    classes =np.argmax(predict)
    text = (word[classes])      
    # if classes==0:
        
    #     print(text)
    print(text)    
    cv2.putText(reta,text,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1.25,(255,255,0),3)
    cv2.imshow('Gesture control', reta)  
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        Response=requests.get("https://iotcloud22.in/1265/post_value.php?value1="+text)
        print("HTTP Response : ",Response.status_code)
        break
video.release()
cv2.destroyAllWindows()