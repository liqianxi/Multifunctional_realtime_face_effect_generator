# Multifunctional_realtime_face_effect_generator

## Introduction
This is an script based on realtime face detection. <br>
First get the landmarks of the face, then use Gaussian blur or thresh effect to process the face area. This can be used when sometimes people in the streaming may not want to show their face. <br>
This script also supports multi-faces in the screen.<br>

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
python3 face_blur.py <option1> <option2>...
```
## Option Usage:
```python3 face_blur.py <shape_group option> <action_group option>```
In shape_group options, you can use only one of```--none```, ```--rect``` and ```--partial```.<br>
In action_group options, you can use only one of ```--gaussian``` and ```--thresh```.<br>
```--mosaic``` option is not finished yet.<br>
You can use ```python3 face_blur.py -h``` to get help with the commands.<br>

## Note:
You should let your terminal use your camera.<br>
There are sometimes when the current frame is not ready, the new window will also not show up.<br> In this case, just rerun the ```python3 face_blur.py <option1> <option2>...``` command.

## Demo:
https://github.com/liqianxi/Multifunctional_realtime_face_effect_generator/blob/main/demo1.mp4 <br>

## Reference:
Many thanks to these articles and tutorials. They inspired me a lot.<br>
https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348 <br>
https://livecodestream.dev/post/detecting-face-features-with-python/<br>
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat<br>
https://blog.csdn.net/qq_41008202/article/details/104397446<br>
https://docs.python.org/3/library/argparse.html#nargs<br>
https://stackoverflow.com/questions/41172918/apply-gaussian-blur-on-a-polygon-using-opencv-and-python<br>
https://blog.csdn.net/m0_38106923/article/details/103836242<br>
