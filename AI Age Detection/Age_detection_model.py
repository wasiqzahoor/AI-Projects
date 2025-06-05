 import cv2
 import numpy as np
 # Load models
 age_proto = 'age_deploy.prototxt'
 age_model = 'age_net.caffemodel'
 face_proto = 'deploy.prototxt'
 face_model = 'res10_300x300_ssd_iter_140000.caffemodel'
 AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
 age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
 face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
 cap = cv2.VideoCapture(0)
 while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = frame[y1:y2, x1:x2]
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
                                              (78.426, 87.768, 114.895), swapRB=False)
            age = AGE_LIST[age_preds[0].argmax()]
            label = f'Age: {age}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
    cv2.imshow("Age Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 cap.release()
cv2.destroyAllWindows()