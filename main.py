from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate


mot_tracker = Sort()

# load models
coco_model = YOLO('./yolov8n.pt')

license_plate_detector = YOLO('./license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]

cars_id = []
cars_number = []

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1 and car_id not in cars_id:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # read license plate number
                license_plate_text,score = read_license_plate(license_plate_crop)

                if license_plate_text is not None:

                    cars_id.append(car_id)
                    cars_number.append((license_plate_text,score))
                    if len(cars_id) > 20:
                        del cars_id[0]
                        del cars_number[0]

            if car_id in cars_id:
                cv2.rectangle(frame,(int(xcar1), int(ycar1)),(int(xcar2), int(ycar2)),(255, 0, 0),12)

        cv2.imshow(frame)
        print(cars_id)
        print(cars_number)
        

cap.release()
cv2.destroyAllWindows()
                                                                    
                                                                    