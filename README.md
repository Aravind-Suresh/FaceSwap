# FaceSwap
A repository which contains source codes for swapping faces in images.

**main.py** Orients input faces onto template faces.

## Trial
First clone the repository. Then,
```
$ python main.py -p /path/to/shape_predictor -t /path/to/template/image -i /path/to/input/image -o /path/to/output/image
```
For example,
```
$ python main.py -p res/shape_predictor_68_face_landmarks.dat -t images/templates/1.png -i images/inputs/2.jpg -o images/outputs/tmpl_1_inp_2.png
```

## Dependencies
* OpenCV ( used 3.0.0 )
* Dlib
* You should also download the shape\_predictor\_68\_face\_landmarks.dat from [here](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2).

## Sample outputs
| Input | Template | Output |
| ----- | -------- | ------ |
|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/inputs/2.jpg" width = "250px" height = "250px"/>|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/templates/1.png" width = "250px" height = "250px" />|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/outputs/tmpl_1_inp_2.png" width = "250px" height = "250px" />|
|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/inputs/4.png" width = "250px" height = "250px"/>|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/templates/2.jpg" width = "250px" height = "250px" />|<img src = "https://raw.githubusercontent.com/Aravind-Suresh/FaceSwap/master/images/outputs/tmpl_2_inp_4.png" width = "250px" height = "250px" />|
