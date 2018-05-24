"""
Application of learned model on the prediction of the attributes from video.
To run see requirements in github.
"""


import cv2
import numpy as np

# Open the input movie file
from keras import optimizers
from keras.models import model_from_json

model_path = "model_for_demo/model"

def load_model(path):
    """
    Load model for predictions from json
    and its weights from .h5 file.
    """
    # load json and create model
    json_file = open(path+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+".h5")
    print("Loaded model from disk")
    return loaded_model


def prediction_to_txt(preds):
    """
    Recodes prediction to string of attribute values.
    :param preds: predictions
    :return: annotated string based on the prediction vector
    """

    hairs = ["black", "blond", "brown", "gray", "other"]
    to_ret = []
    for i in range(len(preds[0])):
        to_ret.append("")

        if np.argmax(preds[0][i]) == 0:
            to_ret[i] += "Attr-"
        else:
            to_ret[i] += "Unat-"

        if np.argmax(preds[1][i]) == 0:
            # to_ret[i] += str(predictions[1][i][0])[1:7] + "-"
            to_ret[i] += "Glass-"
        else:
            # to_ret[i] += str(predictions[1][i][1])[1:7] + "-"
            to_ret[i] += "No gl.-"

        if np.argmax(preds[2][i]) == 0:
            to_ret[i] += "Male-"
        else:
            to_ret[i] += "Female-"

        if np.argmax(preds[3][i]) == 0:
            to_ret[i] += "Smile-"
        else:
            to_ret[i] += "No smile-"

        last = np.argmax(preds[4][i])
        to_ret[i] += hairs[last]

    return to_ret


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.75, thickness=1):
    """
    Support function for drawing text label into the frame.
    :param image: frame to write to
    :param point: (x,y) tuple position of starting point of text in the frame
    :param label: text to write
    :param font: text font
    :param font_scale: text size
    :param thickness: boldness of the text
    :return:
    """
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, (x, y ), font, font_scale, (255, 255, 255), thickness)


def crop_face(imgarray, section, margin=25, size=100):
    """
    Crops face from the frame based on boundig box
    :param imgarray: full image frame
    :param section: face detected area (x, y, w, h) - bounding box
    :param margin: add some margin to the face detected area to include a full head
    :param size: the result image resolution with be (size x size)
    :return: resized image in numpy array with shape (size x size x 3)
    """
    img_h, img_w, _ = imgarray.shape
    if section is None:
        section = [0, 0, img_w, img_h]
    (x, y, w, h) = section
    margin = int(min(w, h) * margin / 100)
    x_a = x - margin
    y_a = y - margin
    x_b = x + w + margin
    y_b = y + h + margin
    if x_a < 0:
        x_b = min(x_b - x_a, img_w - 1)
        x_a = 0
    if y_a < 0:
        y_b = min(y_b - y_a, img_h - 1)
        y_a = 0
    if x_b > img_w:
        x_a = max(x_a - (x_b - img_w), 0)
        x_b = img_w
    if y_b > img_h:
        y_a = max(y_a - (y_b - img_h), 0)
        y_b = img_h
    cropped = imgarray[y_a: y_b, x_a: x_b]
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("Debug", resized_img)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


if __name__ == "__main__":
    input_movie = cv2.VideoCapture("test.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade_file_src = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_file_src)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    face_size = 100
    frame_cnt = 0

    model = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    initial_face = None
    # infinite loop, break by key ESC
    while True:
        # Capture frame-by-frame
        ret, frame = input_movie.read()
        if frame is None:
            break
        faces = []
        frame_cnt += 1
        print(frame_cnt)

        # if frame_cnt < 160:
        #     continue

        if initial_face is not None:
            face_img, cropped = crop_face(frame, initial_face, margin=55, size=face_size)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30))

            if len(faces) >= 1:
                faces = []
                faces.append(initial_face)
        if len(faces) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30))
            if len(faces) >= 1:
                initial_face = faces[0]

        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), face_size, face_size, 3))
        # face_imgs = []
        for i, face in enumerate(faces):
            face_img, cropped = crop_face(frame, face, margin=25, size=face_size)
            (x, y, w, h) = cropped
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i, :, :, :] = face_img
            # face_imgs.append([face_img])

        if len(face_imgs) > 0:
            # predict attributes of the detected faces
            predictions = model.predict(np.expand_dims(face_img,axis=0))
            # cv2.imshow("aaa", face_img)
            # cv2.imwrite("frames/tframe%d.jpg" % frame_cnt, face_img)
            labels = prediction_to_txt(predictions)
            # draw results
            for i, face in enumerate(labels):
                label = labels[i]
                draw_label(frame, ((x, y+h)), label)
        # cv2.imshow('Keras Faces', frame)
        out.write(frame)
        if cv2.waitKey(5) == 27:  # ESC key press
            break

    # All done!
    input_movie.release()
    out.release()
    cv2.destroyAllWindows()
