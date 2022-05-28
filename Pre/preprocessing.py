import copy
import itertools
class Preprocess:
    def __init__(self,image,point_history,landmark_list,**kwargs):
        self.image = image
        self.point_history = point_history
        self.landmark_list = landmark_list

    def normalize_(self,n):
        return n / self.max_value
    def pre_process_landmark(self):
        temp_landmark_list = copy.deepcopy(self.landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        self.max_value = max(list(map(abs, temp_landmark_list)))
        temp_landmark_list = list(map(self.normalize_, temp_landmark_list))
        return temp_landmark_list


    def pre_process_point_history(self):
        image_width, image_height = self.image.shape[1], self.image.shape[0]

        temp_point_history = copy.deepcopy(self.point_history)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] -
                                            base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] -
                                            base_y) / image_height

        # Convert to a one-dimensional list
        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history