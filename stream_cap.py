
import cv2
import os
import numpy as np

# Open the input movie file
from keras import optimizers

from CNN import load_model
from main_training import model_path

hairs = ["black hair", "blond hair", "brown hair", "gray hair", "other"]

def prediction_to_txt(predictions):
    to_ret = []
    # print(predictions)
    for i in range(len(predictions[0])):
        to_ret.append("")

        if np.argmax(predictions[0][i]) == 0:
            to_ret[i] += "A-"
        else:
            to_ret[i] += "U-"

        if np.argmax(predictions[1][i]) == 0:
            to_ret[i] += "G-"
        else:
            to_ret[i] += "NG-"

        if np.argmax(predictions[2][i]) == 0:
            to_ret[i] += "M-"
        else:
            to_ret[i] += "F-"

        if np.argmax(predictions[3][i]) == 0:
            to_ret[i] += "S-"
        else:
            to_ret[i] += "NS-"

        last = np.argmax(predictions[4][i])
        to_ret[i] += hairs[last]

    return to_ret


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def crop_face(imgarray, section, margin=25, size=100):
    """
    :param imgarray: full image
    :param section: face detected area (x, y, w, h)
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
    resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)




if __name__ == "__main__":
    input_movie = cv2.VideoCapture("test5.mp4")
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade_file_src = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_file_src)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0
    face_size = 100

    current_path = os.getcwd()

    model, vars_dict = load_model(model_path)
    opt = optimizers.Adam(lr=0.0000015)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])


    # infinite loop, break by key ESC
    while True:
        # Capture frame-by-frame
        ret, frame = input_movie.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30),
            # minNeighbors=10,
            # minSize=(face_size, face_size)
        )
        # if len(faces) > 1:
        #     tmp = []
        #     tmp.append(faces[0])
        #     faces = tmp
        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), face_size, face_size, 3))
        for i, face in enumerate(faces):
            face_img, cropped = crop_face(frame, face, margin=35, size=face_size)
            (x, y, w, h) = cropped
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            face_imgs[i, :, :, :] = face_img

        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            predictions = model.predict(face_imgs)
            labels = prediction_to_txt(predictions)
            # draw results
            for i, face in enumerate(faces):
                label = labels[i]
                draw_label(frame, (face[0], face[1]), label)
        cv2.imshow('Keras Faces', frame)
        if cv2.waitKey(5) == 27:  # ESC key press
            break

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()()
