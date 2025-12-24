import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk
import PIL.Image
from itertools import count
import string
from tkinter import *
import time
import multiprocessing as mproc
#----------
from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file
from src.config import SEQ_LEN, THRESH_HOLD
import numpy as np
import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

s2p_map = {k.lower(): v for k, v in load_json_file("src/sign_to_prediction_index_map.json").items()}
p2s_map = {v: k for k, v in load_json_file("src/sign_to_prediction_index_map.json").items()}
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

models_path = [
    './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
]
models = [get_model() for _ in models_path]

# Load weights from the weights file.
for model, path in zip(models, models_path):
    model.load_weights(path)


def real_time_asl():
    """
    Perform real-time ASL recognition using webcam feed.

    This function initializes the required objects and variables, captures frames from the webcam, processes them for hand tracking and landmark extraction, and performs ASL recognition on a sequence of landmarks.

    Args:
        None

    Returns:
        None
    """
    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []
    cap = cv2.VideoCapture(0)

    start = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # The main loop for the mediapipe detection.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()

            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)

            try:
                landmarks = extract_coordinates(results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))
            sequence_data.append(landmarks)

            sign = ""

            # Generate the prediction for the given sequence data.
            if len(sequence_data) % SEQ_LEN == 0:
                prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]

                if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                    sign = np.argmax(prediction.numpy(), axis=-1)

                sequence_data = []

            image = cv2.flip(image, 1)

            cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            image = cv2.flip(image, 1)

            # Insert the sign in the result set if sign is not empty.
            if sign != "" and decoder(sign) not in res:
                res.insert(0, decoder(sign))

            # Get the height and width of the image
            height, width = image.shape[0], image.shape[1]

            # Create a white column
            white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

            # Flip the image vertically
            image = cv2.flip(image, 1)

            # Concatenate the white column to the image
            image = np.concatenate((white_column, image), axis=0)

            cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Webcam Feed', image)

            # Wait for a key to be pressed.
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty('Webcam Feed', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
#--------

asl_process = None


def start_asl_process():
    global asl_process
    if asl_process is not None and asl_process.is_alive():
        return
    asl_process = mproc.Process(target=real_time_asl, daemon=True)
    asl_process.start()


try:
    import Tkinter as tk
except:
    import tkinter as tk
import numpy as np

image_x, image_y = 64, 64
from keras.models import load_model

classifier = load_model('model.h5')


def give_char():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('tmp1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)
    chars = "ABCDEFGHIJKMNOPQRSTUVWXYZ"
    indx = np.argmax(result[0])
    print(indx)
    return (chars[indx])


def check_sim(i, file_map):
    for item in file_map:
        for word in file_map[item]:
            if (i == word):
                return 1, item
    return -1, ""


op_dest = "/Users/Utkarsh/Desktop/major project/sign language detection system/1/filtered_data/"
alpha_dest = "/Users/Utkarsh/Desktop/major project/sign language detection system/1/alphabet/"
dirListing = os.listdir(op_dest)
editFiles = []
for item in dirListing:
    if ".webp" in item:
        editFiles.append(item)

file_map = {}
for i in editFiles:
    tmp = i.replace(".webp", "")
    # print(tmp)
    tmp = tmp.split()
    file_map[i] = tmp


def func(a):
    all_frames = []
    final = PIL.Image.new('RGB', (380, 260))
    words = a.split()
    for i in words:
        flag, sim = check_sim(i, file_map)
        if (flag == -1):
            for j in i:
                print(j)
                im = PIL.Image.open(alpha_dest + str(j).lower() + "_small.gif")
                frameCnt = im.n_frames
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)
                    im.save("tmp.png")
                    img = cv2.imread("tmp.png")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (380, 260))
                    im_arr = PIL.Image.fromarray(img)
                    for itr in range(15):
                        all_frames.append(im_arr)
        else:
            print(sim)
            im = PIL.Image.open(op_dest + sim)
            im.info.pop('background', None)
            im.save('tmp.gif', 'gif', save_all=True)
            im = PIL.Image.open("tmp.gif")
            frameCnt = im.n_frames
            for frame_cnt in range(frameCnt):
                im.seek(frame_cnt)
                im.save("tmp.png")
                img = cv2.imread("tmp.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (380, 260))
                im_arr = PIL.Image.fromarray(img)
                all_frames.append(im_arr)
    final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
    return all_frames


img_counter = 0
img_text = ''


class Tk_Manage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, VtoS, StoV):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Two Way Sign Langage Translator", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        button = tk.Button(self, text="Text to Sign", command=lambda: controller.show_frame(VtoS))
        button.pack()
        button2 = tk.Button(self, text="Sign to Text", command=start_asl_process)
        button2.pack()
        load = PIL.Image.open("Two Way Sign Language Translator.png")
        load = load.resize((620, 450))
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=100, y=200)


class VtoS(tk.Frame):
    def __init__(self, parent, controller):
        cnt = 0
        gif_frames = []
        inputtxt = None
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Text to Sign", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        gif_box = tk.Label(self)

        button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = tk.Button(self, text="Sign to Text", command=start_asl_process)
        button2.pack()

        def gif_stream():
            global cnt
            global gif_frames
            if (cnt == len(gif_frames)):
                return
            img = gif_frames[cnt]
            cnt += 1
            imgtk = ImageTk.PhotoImage(image=img)
            gif_box.imgtk = imgtk
            gif_box.configure(image=imgtk)
            gif_box.after(50, gif_stream)

        def Take_input():
            INPUT = inputtxt.get("1.0", "end-1c")
            print(INPUT)
            global gif_frames
            gif_frames = func(INPUT)
            global cnt
            cnt = 0
            gif_stream()
            gif_box.place(x=400, y=160)

        l = tk.Label(self, text="Enter Text:")
        inputtxt = tk.Text(self, height=4, width=25)
        Display = tk.Button(self, height=2, width=20, text="Convert", command=lambda: Take_input())
        l.place(x=50, y=160)
        inputtxt.place(x=50, y=250)
        Display.pack()



class StoV(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Sign to Text", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = tk.Button(self, text="Text to Sign", command=lambda: controller.show_frame(VtoS))
        button2.pack()
        disp_txt = tk.Text(self, height=4, width=25)

        def start_video():
            video_frame = tk.Label(self)
            cam = cv2.VideoCapture(0)

            global img_counter
            img_counter = 0
            global img_text
            img_text = ''

            def video_stream():
                global img_text
                global img_counter
                if (img_counter > 200):
                    return None
                img_counter += 1
                ret, frame = cam.read()
                frame = cv2.flip(frame, 1)
                img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
                lower_blue = np.array([35, 10, 0])
                upper_blue = np.array([160, 230, 255])
                imcrop = img[102:298, 427:623]
                hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
                img_name = "tmp1.png"
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                tmp_text = img_text[0:]
                img_text = give_char()
                if (tmp_text != img_text):
                    print(tmp_text)
                    disp_txt.insert(END, tmp_text)
                img = PIL.Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                video_frame.imgtk = imgtk
                video_frame.configure(image=imgtk)
                video_frame.after(1, video_stream)

            video_stream()
            disp_txt.pack()
            video_frame.pack()

        start_vid = tk.Button(self, height=2, width=20, text="Start Video", command=lambda: start_video())
        start_vid.pack()


if __name__ == "__main__":
    mproc.freeze_support()
    app = Tk_Manage()
    app.geometry("800x750")
    app.mainloop()
