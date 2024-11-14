import logging

import numpy as np
import glob
import os
import cv2


class StereoCalibrator:
    def __init__(self, config):
        self.chessboard_size = tuple(config['chessboard_size'])
        self.square_size = config['square_size']
        self.logger = logging.getLogger('autonomous_perception.calibration')


    def calibrate(self, calibration_images):
        self.logger.info("Starting stereo calibration")

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        # Separate lists for object points and image points for each camera
        objpoints_left = []  # 3d points in real world space for the left camera
        objpoints_right = []  # 3d points in real world space for the right camera
        imgpoints_left = []   # 2d points in the left camera image plane
        imgpoints_right = []  # 2d points in the right camera image plane

        # Loop over left (image_02) and right (image_03) folders, with separate lists
        for folder, objpoints, imgpoints in zip(
            ['image_02', 'image_03'],
            [objpoints_left, objpoints_right],
            [imgpoints_left, imgpoints_right]
        ):
            image_path = os.path.join(calibration_images, folder, '*')
            images = glob.glob(image_path)

            # Process each image in the folder
            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_size[0], self.chessboard_size[1]))

                # If found, add object points and image points to the correct lists
                if ret:
                    objpoints.append(objp)  
                    imgpoints.append(corners)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (self.chessboard_size[0], self.chessboard_size[1]), corners, ret)
                    plt.imshow(img)
                    plt.show()

        # Calibrate each camera separately
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints_left, imgpoints_left, gray.shape[::-1], None, None
        )
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints_right, imgpoints_right, gray.shape[::-1], None, None
        )

        # Calculate optimal camera matrix for undistortion
        h, w = gray.shape[:2]
        newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(mtx_left, dist_left, (w, h), 1, (w, h))
        newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(mtx_right, dist_right, (w, h), 1, (w, h))

        self.logger.info("Stereo calibration completed")
        
        return {
            "left_camera": {"matrix": mtx_left, "distortion": dist_left, "rvecs": rvecs_left, "tvecs": tvecs_left},
            "right_camera": {"matrix": mtx_right, "distortion": dist_right, "rvecs": rvecs_right, "tvecs": tvecs_right},
        }

        
 
