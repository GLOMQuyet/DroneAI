import csv

class Login:
    def __init__(self,number, mode, landmark_list, point_history_list,**kwargs):
        self.number = number
        self.mode = mode
        self.landmark_list = landmark_list
        self.point_history_list = point_history_list

    def logging_csv(self):
        if self.mode == 0:
            pass
        if self.mode == 1 and (0 <= self.number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.number, *self.landmark_list])
        if self.mode == 2 and (0 <= self.number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.number, *self.point_history_list])
        return