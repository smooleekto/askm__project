from PIL import Image
import numpy as np
import numpy as np
import os
import cv2
from keras.models import load_model
import sys
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PATH = os.path.dirname(os.path.abspath(__file__))
gender_model = load_model(f'{PATH}/models/gender_model.h5')
age_model = load_model(f'{PATH}/models/age_model.h5')


def face_find(user_image):
    face_cascade = cv2.CascadeClassifier(f'{PATH}/models/haarcascade_frontalface_default.xml')
    img = cv2.imread(user_image)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5)
    cords = []
    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x-15, y-15), (x+w+15, y+h+15), (0, 255, 0), 2)
        crop_img = img[y:y+h, x:x+w]
        cords.append((x, y-30))
        cv2.imwrite(f"{PATH}/output/{user_image.split('/')[-1].split('.')[0]}-{index+1}.jpg", crop_img)
    return [img, cords]
    

def process_and_predict(file):
    print(file.split('/')[-1])
    cv2img = face_find(file)
    current_photo = []
    for (root,dirs,files) in os.walk(f'{PATH}/output', topdown=True):
        i = 0
        for face in files: 
            if file.split('/')[-1].split('.')[0]+'-' in face:
                current_photo.append(face)
                i+=1
        if i == 0:
            print('На фото не знайдені обличчя')
            return 0
    for item in current_photo:
        img = Image.open(f'{PATH}/output/{item}')
        width, height = img.size
        if width == height:
            img = img.resize((200,200), Image.Resampling.LANCZOS)
        else:
            if width > height:
                left = width/2 - height/2
                right = width/2 + height/2
                top = 0
                bottom = height
                img = img.crop((left,top,right,bottom))
                img = img.resize((200,200), Image.Resampling.LANCZOS)
            else:
                left = 0
                right = width
                top = 0
                bottom = width
                img = img.crop((left,top,right,bottom))
                img = img.resize((200,200), Image.Resampling.LANCZOS)

        matrix = np.asarray(img)
        matrix = matrix.astype('float32')
        matrix /= 255.0

        matrix = matrix.reshape(-1, 200, 200, 3)
        age = age_model.predict(matrix)
        gender = np.round(gender_model.predict(matrix))
        if gender == 0:
            gender = 'male'
        elif gender == 1:
            gender = 'female'
        for item in cv2img[1]:
            cv2.putText(cv2img[0], f'{gender}, {int(age)} y.o.', item, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 0, 158), thickness=2)
        print('Вік:', int(age), '\nСтать:', gender)
    cv2.imshow('Age prediction', cv2img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"{PATH}/output/{file.split('/')[-1].split('.')[0]}__processed.jpg", cv2img[0])

def pretrained_predict():
    for (root,dirs,files) in os.walk(f'{PATH}/input', topdown=True):
        if len(files)==0:
            print('В директорії input відсутні файли')
        for file in files:
            process_and_predict(f'{PATH}/input/{file}')


def new_model():
    images = []
    ages = []
    genders = []
    epoch = int(input('Введіть кількість епох: '))
    learning_rate = float(input('Введіть швидкість навчання: '))
    tf.keras.backend.clear_session()
    if len(os.listdir(f'{PATH}/dataset/'))==0:
        print('Датасет не знайдений')
        return 0
    for i in os.listdir(f'{PATH}/dataset/')[0:8000]:
        split = i.split('_')
        ages.append(int(split[0]))
        genders.append(int(split[1]))
        images.append(Image.open('dataset/' + i))
    images = pd.Series(list(images), name = 'Images')
    ages = pd.Series(list(ages), name = 'Ages')
    genders = pd.Series(list(genders), name = 'Genders')
    df = pd.concat([images, ages, genders], axis=1)

    x = []
    y = []
    for i in range(len(df)):
        df['Images'].iloc[i] = df['Images'].iloc[i].resize((200,200), Image.Resampling.LANCZOS)
        matrix = np.asarray(df['Images'].iloc[i])
        x.append(matrix)
        agegen = [int(df['Ages'].iloc[i]), int(df['Genders'].iloc[i])]
        y.append(agegen)
    x = np.array(x)
    y_age = df['Ages']
    y_gender = df['Genders']
    x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x, y_age, test_size=0.2)
    x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x, y_gender, test_size=0.2)
    age_model = Sequential()
    age_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    age_model.add(MaxPooling2D((2,2)))
    age_model.add(Conv2D(64, (3,3), activation='relu'))
    age_model.add(MaxPooling2D((2,2)))
    age_model.add(Conv2D(128, (3,3), activation='relu'))
    age_model.add(MaxPooling2D((2,2)))
    age_model.add(Flatten())
    age_model.add(Dense(64, activation='relu'))
    age_model.add(Dropout(0.5))
    age_model.add(Dense(1, activation='relu'))
    age_model.compile(loss='mean_squared_error', 
                optimizer=optimizers.Adam(learning_rate=learning_rate))
    gender_model = Sequential()
    gender_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)))
    gender_model.add(MaxPooling2D((2,2)))
    gender_model.add(Conv2D(64, (3,3), activation='relu'))
    gender_model.add(MaxPooling2D((2,2)))
    gender_model.add(Conv2D(128, (3,3), activation='relu'))
    gender_model.add(MaxPooling2D((2,2)))
    gender_model.add(Flatten())
    gender_model.add(Dense(64, activation='relu'))
    gender_model.add(Dropout(0.5))
    gender_model.add(Dense(1, activation='sigmoid'))

    gender_model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
    standart = ImageDataGenerator(rescale=1./255., width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
    train1 = standart.flow(x_train_age, y_train_age, batch_size=64)
    standart1 = ImageDataGenerator(rescale=1./255)
    test1 = standart1.flow(
            x_test_age, y_test_age,
            batch_size=64)
    age_fit = age_model.fit(train1, epochs=epoch, shuffle=True, validation_data=test1)
    age_model.save('{PATH}/models/age_model.h5')


    train2 = standart.flow(x_train_gender, y_train_gender, batch_size=64)

    test2 = standart1.flow(
            x_test_gender, y_test_gender,
            batch_size=64)

    model_fit = gender_model.fit(train2, epochs=epoch, shuffle=True, validation_data=test2)
    gender_model.save(f'{PATH}/models/gender_model.h5')
    print('Моделі збережено!')

def info():
    print('Зверніть увагу, назви фото в датасеті повинні відповідати наступному шаблону: "Вiк_Стать_Назва.jpg"')
    images = []
    ages = []
    genders = []
    if len(os.listdir(f'{PATH}/dataset/'))==0:
        print('Датасет не знайдений')
        return 0
    for i in os.listdir(f'{PATH}/dataset/')[0:8000]:
        split = i.split('_')
        ages.append(int(split[0]))
        genders.append(int(split[1]))
        images.append(Image.open('dataset/' + i))
    images = pd.Series(list(images), name = 'Images')
    ages = pd.Series(list(ages), name = 'Ages')
    genders = pd.Series(list(genders), name = 'Genders')
    df = pd.concat([images, ages, genders], axis=1)
    plt.figure(figsize=(15, 15))
    plt.bar(['Male', 'Female'], [len(df[df['Genders'] == 0]), len(df[df['Genders'] == 1])])
    plt.show()
    plt.plot(df['Ages'].sort_values().unique().tolist(), df['Ages'].sort_values().value_counts().tolist())
    plt.show()

def menu():
    print('Меню')
    print('1. Використати готову модель')
    print('2. Створити нову модель')
    print('3. Інформація про датасет')
    print('4. Вихід')
    ans = input('Оберіть варіант:')
    while ans not in ['1', '2', '3', '4']:
        ans = input('Оберіть варіант:')
    if ans == '1':
        pretrained_predict()
        menu()
    if ans == '2':
        new_model()
        menu()
    if ans == '3':
        info()
        menu()
    if ans == '4':
        sys.exit(0)
if __name__ == '__main__':
    menu()
