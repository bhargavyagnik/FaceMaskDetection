import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('saved_model/model_3.h5')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(1)

labels_dict = {0: 'with_mask', 1: 'without_mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

while (True):

    ret, img = source.read()
    faces = face_clsfr.detectMultiScale(img)
    print(img.shape)
    for x, y, w, h in faces:
        face_img = img[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (128, 128))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 128, 128, 3))
        result = model.predict(reshaped)
        print(result)
        label=int(result.round().flatten())
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(
            img, labels_dict[label],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
source.release()