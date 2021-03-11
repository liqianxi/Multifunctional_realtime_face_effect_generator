# Multifunctional_realtime_face_effect_generator



## How to use?
First, go to https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat.<br> 
Download the file and place it at our root folder.<br>
It's a pretrained model for face detection.

Second, download necessary Python3 libraries. 
```
pip install dlib
pip install numpy
pip install opencv-python
```
Run the following commands to get started.
```
git clone https://github.com/liqianxi/Multifunctional_realtime_face_effect_generator.git face_effect_generator
cd face_effect_generator
python3 face_blur.py
```
## Note:
You should let your terminal use your camera.<br>
There are sometimes when the current frame is not ready, the new window will also not show up.<br> In this case, just rerun the ```python3 face_blur.py``` command.
