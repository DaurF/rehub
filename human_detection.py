from ultralytics import YOLO
import cv2
import os

# video_path_A = 'input_video_from_Maulen/22.mp4'
video_path_B = 'input_video_from_Dauren/three_correct.mp4'

frames_folder = 'frame_from_8_video'
os.makedirs(frames_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path_B)

model = YOLO('yolov8x.pt')

cnt = 0
saved_cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if cnt % 3 == 0:
        frame_height, frame_width = frame.shape[:2]
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:  # Class ID 0 is 'person' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw bounding box on the frame (optional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Person: {box.conf.item():.2f}'
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Save the detected person as an image
                    person_crop = frame[y1:y2, x1:x2]
                    crop_file_name = os.path.join(frames_folder, f'person10_{saved_cnt}.jpg')
                    cv2.imwrite(crop_file_name, person_crop)
                    saved_cnt += 1

    cnt += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {saved_cnt} full-body frames to {frames_folder}.")
