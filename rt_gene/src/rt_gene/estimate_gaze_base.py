# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os
import cv2
import math
import numpy as np
from tqdm import tqdm


from . gaze_tools import get_endpoint, get_euler_from_phi_theta



#AARAV - function to obtain angle of the line points
def getAngleBetweenPoints(x_orig, y_orig, x_landmark, y_landmark):
    deltaY = y_landmark - y_orig
    deltaX = x_landmark - x_orig
    return math.degrees(math.atan2(deltaY, deltaX))
    #AARAV - We can comment line above & uncomment this line to get angle between 0 to 360 degrees
    # return (math.degrees(math.atan2(deltaY, deltaX)) + 360) % 360 # do this for 0 to 360
    
    

class GazeEstimatorBase(object):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation."""
    def __init__(self, device_id_gaze, model_files):
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))
        self.device_id_gazeestimation = device_id_gaze
        self.model_files = model_files

        if not isinstance(model_files, list):
            self.model_files = [model_files]

        if len(self.model_files) == 1:
            self._gaze_offset = 0.11
        else:
            self._gaze_offset = 0.0

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        pass

    def input_from_image(self, cv_image):
        pass

    @staticmethod
    def visualize_eye_result(eye_image, est_gaze):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(eye_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

        endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50) # est_gaze[0]=theta, est_gaze[1]=pie hjsong

        #AARAV - calling getAngleBetweenpoints - This function return angle betwee -180 to 180 degrees
        #degree_angle = getAngleBetweenPoints(center_x, center_y, endpoint_x, endpoint_y)   
        #tqdm.write('__________-------------- Degree '+str(degree_angle))

        #hjsong  
        #tqdm.write('__________-------------- Gaze Degree (in theta pie) '+str(est_gaze) )
        theta = est_gaze[0]
        phi = est_gaze[1]

        Degrees=get_euler_from_phi_theta(phi, theta)
        #roll pitch yaw
        E_Degrees = [0,0,0]
        E_Degrees[0] = -math.degrees(math.asin(math.sin(Degrees[0]))) #roll
        E_Degrees[1] = math.degrees(math.asin(math.sin(Degrees[1])))  #pitch
        E_Degrees[2] = math.degrees(math.asin(math.sin(Degrees[2])))  #yaw

        #tqdm.write('__________-------------- Gaze Degree (in euler angles) '+str(E_Degrees) )
        #hjsong

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Adding angle value on the image
        #cv2.putText(output_image, str(degree_angle), (int(endpoint_x), int(endpoint_y)), font, .4, 255, 1, cv2.LINE_AA)
        
        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))

        #AARAV also return the degree
        #return output_image, degree_angle
        return output_image, E_Degrees  #hjsong  yaw  @Aarav, you can try others like E_Degrees[0], E_Degrees[1] to see if it does make a sense