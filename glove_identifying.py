import os
import sys

os.chdir("BaseballCV")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../BaseballCV')))

from BaseballCV.scripts.load_tools import LoadTools
from BaseballCV.scripts.savant_scraper import BaseballSavVideoScraper

from ultralytics import YOLO
import cv2 
from tqdm import tqdm
import json

scraper = BaseballSavVideoScraper()

pitch_data = scraper.run_statcast_pull_scraper(start_date='2024-08-02', 
                                               end_date='2024-08-02', 
                                               download_folder=r'../videos', 
                                               team='BOS')

load_tools = LoadTools()
model_weights = load_tools.load_model(model_alias='glove_tracking')
model = YOLO(model_weights)

pitch_frames = {}
min_frame = 50
max_frame = 200

for video_path in tqdm(os.listdir(r'../videos')[:5]):

    pitch_id = video_path[7:-4]

    cap = cv2.VideoCapture(r'../videos/' + video_path)

    heights = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    for frame_num in range(min_frame, max_frame):
        
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, classes=[0, 1], device='cuda', verbose=False)
        conf = 0

        if len(results) > 0:

            glove_box, plate_box = None, None

            for box in results[0].boxes:
                if box.cls == 0:
                    glove_box = box
                    break
            
            for box in results[0].boxes:
                if box.cls == 1:
                    plate_box = box
                    break

            if glove_box is None or plate_box is None:
                continue

            glove_x1, glove_y1, glove_x2, glove_y2 = glove_box.xyxy[0]
            glove_x1, glove_y1, glove_x2, glove_y2 = int(glove_x1), int(glove_y1), int(glove_x2), int(glove_y2)
            glove_conf = float(glove_box.conf[0])

            plate_x1, plate_y1, plate_x2, plate_y2 = plate_box.xyxy[0]
            plate_x1, plate_y1, plate_x2, plate_y2 = int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)
            plate_conf = float(plate_box.conf[0])

            frame_attrs = {'glove':{'x1':glove_x1, 'y1':glove_y1, 'x2':glove_x2, 'y2':glove_y2, 'conf':glove_conf},
                           'plate':{'x1':plate_x1, 'y1':plate_y1, 'x2':plate_x2, 'y2':plate_y2, 'conf':plate_conf}, 
                           'frame':frame_num}

            if glove_conf > 0.75:
                heights[frame_num] = frame_attrs

    try:
        pitch_frames[pitch_id] = heights[min(heights, key=lambda k: heights[k]['glove']['y1'])]
    except:
        pitch_frames[pitch_id] = {'glove':{'x1':-1, 'y1':-1, 'x2':-1, 'y2':-1, 'conf':-1},
                                  'plate':{'x1':-1, 'y1':-1, 'x2':-1, 'y2':-1, 'conf':-1},
                                  'frame':-1}
        
with open('../pitch_frames.json', 'w') as f:
    json.dump(pitch_frames, f)

scraper.cleanup_savant_videos(r'../videos')
