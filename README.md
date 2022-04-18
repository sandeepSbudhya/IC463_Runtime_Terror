# Accident Detection using Tensorflow, Pytorch, OpenCV and YOLOv3.
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/slide1.PNG?token=AHGCFGC26O6IC3ODZ7X42HK7GD2X4)

----

## Installation Guide 

1. Clone the repository

### Server 

2. Install all dependencies by running

``` 
pip install requirements.txt
```

3. Download Yolov3.weights from [here](https://pjreddie.com/media/files/yolov3.weights) and place it in folder named 'server/cfg/'

### Database Configuration

4. Download firebase credentials from your firebase database and store it in 'server/firebase_credentials.json'.

5. Change lines 68-76 in 'server/server.py' to match your firebase configuration.

6. Change firebase configuration in 'src/firebase/init.js' to match your firebase configuration.


### Web-Application


7. Install node modules
```
npm install
```
rojec
8. Compiles and minifies for production and is stored in 'dist/'
```
npm run build
```

---

## Starting the  Server 

9. Navigate to 'server/' and run the command
```
python server.py
```

---
## Description
### Demo
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/demo.gif?token=AHGCFGCNJPVZPT3SAP47JKK7GD6OW)

![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/whatsapp_demo.gif?token=AHGCFGFLTJ3XWRUBX2AN5SC7GD6QY)
### Approach
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/approach.PNG?token=AHGCFGFVJSSB57RZ7S2BA527GD4ZU)
### Statistics on accidents in India
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/slide2.PNG?token=AHGCFGEI35FOHND22DQ3EXK7GD2YO)

### Workflow
![image alt](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/slide3.PNG?token=AHGCFGDMPBSXPNUUSBCAHUC7GD2Y2)

### Collision and Proximity detection algorithm
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/slide4.PNG?token=AHGCFGHK3R7YMUXGIWKUMVC7GD3IQ)


### Feature extraction and binary classification model using tensorflow
![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/model%20flow.PNG?token=AHGCFGCZ32XV5J3VHNL6XVC7GD464)

![](https://raw.githubusercontent.com/iamrgm/IC463_Runtime_Terror/master/Images/slide5.PNG?token=AHGCFGGPQMZWIYWC42BAW527GEK34)






