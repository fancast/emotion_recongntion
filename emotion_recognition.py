import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import tqdm

face_data = pd.read_csv("/data/faces/fer2013/fer2013.csv")
face_data.head()


labels = []
usage = []

for i in face_data["emotion"]:
    labels.append(i)
    
for i in face_data["Usage"]:
    usage.append(i)

count0 = 0

for i, j, k in tqdm(zip(face_data["emotion"], face_data["pixels"], face_data["Usage"])):
    pixel = []
    pixels = j.split(' ')
    for m in pixels:
        value = float(m)
        pixel.append(value)
    pixel = np.array(pixel)
    image = pixel.reshape(48, 48)
    
    if k == "Training":
        if not os.path.exists("train"):
            os.mkdir("train")
        if i == 0:
            if not os.path.exists("train/Angry"):
                os.mkdir("train/Angry")
                path = "train/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "train/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
    else:
        if not os.path.exists("validation"):
            os.mkdir("validation")
        if i == 0:
            if not os.path.exists("validation/Angry"):
                os.mkdir("validation/Angry")
                path = "validation/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1
            else:
                path = "validation/Angry/" + str(count0) + ".jpg"
                cv2.imwrite(path , image)
                count0 += 1


train_counts = []
train_dir = [face_data]

for folder in train_dir:
    class_path = train_directory + folder + "/"
    list_train = []
    count = 0
    for file in os.listdir(class_path):
        count +=1
    
    train_counts.append(count)
    
train_counts


plt.bar(classes, train_counts, width=0.5)
plt.title("Bar Graph of Train Data")
plt.xlabel("Classes")
plt.ylabel("Counts")

plt.scatter(classes, train_counts)
plt.plot(classes, train_counts, '-o')
plt.show()
