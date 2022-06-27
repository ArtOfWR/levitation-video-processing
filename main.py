import cv2
import numpy as np
import pandas as pd
from pathlib import Path


show_images = False
display_data = True


if __name__ == '__main__':
    video_folder = Path('D:/SVN/Akhmerov/Levitation/Video_16_06_2022')
    video_names = [Path('Basler_acA1300-200uc__22030382__20220616_142333413.mp4')]

    for video_name in video_names:
        cap = cv2.VideoCapture(str(video_folder / video_name))
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Processing video: ' + str(video_folder / video_name) + f' that consists of {video_length} frames')

        frame_counter = 0
        display_counter = 0
        result = np.zeros(video_length)
        while cap.isOpened():
            ret, src = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            th, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            if show_images:
                cv2.imshow('Image', binary)
                cv2.waitKey(1)

            rows, cols = binary.shape
            row_step = 10

            x_sum = 0
            x_count = 0
            for i in range(0, rows, row_step):
                col_sum = 0
                col_count = 0
                for j in range(cols):
                    if binary[i, j] > 0:
                        col_sum += j
                        col_count += 1
                x_sum += col_sum/col_count
                x_count += 1
            result[frame_counter] = x_sum/x_count
            if display_data:
                print(f'Frame: {frame_counter}, x: {result[frame_counter]}')
            frame_counter += 1


        data_frame = pd.DataFrame(result, index=range(frame_counter), columns=['ResultPix'])
        data_frame.to_excel(video_folder / Path(f'result({video_name}).xlsx'))
