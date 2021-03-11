import argparse
import copy
import cv2
import dlib
import numpy as np
import sys


def parse_arguments():
    rect = True
    no_op = True
    action = "gaussian"
    if len(sys.argv) > 1:
        # Means we have options.
        no_op = False
        option_list = sys.argv[1:]
        parser = argparse.ArgumentParser()
        shape_group = parser.add_mutually_exclusive_group()
        shape_group.add_argument('--rect', 
                                 action="store_true", 
                                 help='Enable full blur on the face area.')
        shape_group.add_argument('--partial', 
                                 action="store_true", 
                                 help='Enable partial blur on the face area.')

        action_group = parser.add_mutually_exclusive_group()

        action_group.add_argument('--gaussian', 
                                  action="store_true", 
                                  help='Specify the blur type to be gaussian.')
        action_group.add_argument('--mosaic', 
                                  action="store_true", 
                                  help='Specify the blur type to be mosaic.')
        action_group.add_argument('--thresh', 
                                  action="store_true", 
                                  help='Let the area color be threshed.')

        args = parser.parse_args(sys.argv[1:])

        if args.partial:
            rect = False

        if args.thresh:
            action = "thresh"

        return (rect, no_op, action)

def get_masked_picture(image_copy, blurred_image, roi_corners):
    # create a mask for the ROI and fill in the ROI with (255,255,255) color
    mask = np.zeros(image_copy.shape, dtype=np.uint8)
    channel_count = image_copy.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # create a mask for everywhere in the original image except the ROI, 
    # (hence mask_inverse)
    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
    # combine all the masks and above images in the following way
    
    final_image = ( cv2.bitwise_and(blurred_image, mask) 
                  + cv2.bitwise_and(image_copy, mask_inverse))
    return final_image

def main():
    rect, no_op, action = parse_arguments()
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # set Width
    cap.set(4, 480) # set Height

    while(True):
        ret, frame = cap.read()
        try:
            temp = frame.all() != None
        except Exception as e:
            print(e)
            
        else:
            if temp:
                if no_op:
                    cv2.imshow('frame', frame)

                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Use detector to find landmarks
                    
                    faces = detector(gray)
                    outer_edge_list = []
                    final_image = frame
                    for face in faces:
                        x1 = face.left() # left point
                        y1 = face.top() # top point
                        x2 = face.right() # right point
                        y2 = face.bottom() # bottom point
                        # Create landmark object
                        landmarks = predictor(image=gray, box=face)
                        if rect:
                            outer_edge_list = [(x1,y1), 
                                               (x1,y2), 
                                               (x2,y2), 
                                               (x2,y1)]

                        else:
                            range_list = [i for i in range(17)] + \
                                         [i for i in range(26,16,-1)]
                            outer_edge_list = [(landmarks.part(n).x, 
                                                landmarks.part(n).y) \
                                                    for n in range_list] 
                              
                        # make an area of picture blur
                        if outer_edge_list == []:
                            continue

                        roi_corners = np.array([outer_edge_list], 
                                               dtype = np.int32)
                        image_copy = copy.deepcopy(final_image)

                        if action == "gaussian":
                            # create a blurred copy of the entire image
                            
                            blurred_image = cv2.GaussianBlur(image_copy,
                                                             (79, 79),
                                                             30)
                            final_image = get_masked_picture(image_copy,
                                                             blurred_image,
                                                             roi_corners)
                            
                        elif action =="thresh":
                            blurred_image = cv2.threshold(frame, 
                                                          60, 
                                                          255, 
                                                          cv2.THRESH_BINARY)[1]
                            final_image = get_masked_picture(image_copy, 
                                                             blurred_image, 
                                                             roi_corners)
                        # show the image
                        cv2.imshow('frame', final_image)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

main()
'''
https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348
https://livecodestream.dev/post/detecting-face-features-with-python/
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
https://blog.csdn.net/qq_41008202/article/details/104397446
https://docs.python.org/3/library/argparse.html#nargs
https://stackoverflow.com/questions/41172918/apply-gaussian-blur-on-a-polygon-using-opencv-and-python
https://blog.csdn.net/m0_38106923/article/details/103836242
'''