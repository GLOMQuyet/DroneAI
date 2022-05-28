import numpy as np
import cv2 as cv
class Calc:
    def __init__(self,image,landmarks):
        self.image = image
        self.landmarks = landmarks

    def calc_bounding_rect(self):
        image_width, image_height = self.image.shape[1], self.image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(self.landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]


    def calc_landmark_list(self):
        image_width, image_height = self.image.shape[1], self.image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(self.landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point