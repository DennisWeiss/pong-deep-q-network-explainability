import matplotlib.pyplot as plt
import gym
import numpy as np
import cv2

# Enter N 3 channel pictures array
# Output: an array shape (84 84 N)
# : 1. resize ==>(84 84 3)[uint 0-255]
#       2. gray   ==> (84 84 1) [uint 0-255]
#       3. norm   ==> (84 84 1) [float32 0.0-1.0]
#       4. concat ===>(84 84 N) [float32 0.0-1.0]
def imgbuffer_process(imgbuffer, out_shape = (80, 64)):
    img_list = []
    for img in imgbuffer:
        tmp = cv2.resize(src=img, dsize=out_shape)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        ## Need to convert data type to 32F
        tmp = cv2.normalize(tmp, tmp, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Expand a dimension
        tmp = np.expand_dims(tmp, len(tmp.shape))
        img_list.append(tmp)
    ret =  np.concatenate(tuple(img_list), axis=2)
    #print('ret_shape = ' + str(ret.shape))
    return ret

def get_image_data(state):
    img_buffer = []
    img_buffer_size = 4

    if len(img_buffer) < img_buffer_size:
        img_buffer.append(state)
        pass
    else:
        img_buffer.pop(0)
        img_buffer.append(state)

    img_input = imgbuffer_process(img_buffer)
    return img_input

