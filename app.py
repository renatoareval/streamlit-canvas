import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas


def load_model():
    try:
        # carregar um modelo pré treinado
        model = tf.keras.models.load_model("modelo_cnn.h5")
    except:
        # carregar o dataset e preparar os dados
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
        test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # construir o modelo
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # compilação do modelo
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # treinando o modelo
        model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

        # salvar o modelo treinado
        model.save('modelo_cnn.h5')

    return model

modelo = load_model()

# app
st.title('Reconhecimento de digitos manuscritos')

# canvas para o  desenho
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150, width=150,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img = img.resize((28, 28), Image.ANTIALIAS)
    img = ImageOps.invert(img)
    img = img.convert("L")
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    prediction = np.argmax(modelo.predict(img), axis=1)
    st.write('Previsão: ', prediction[0])

st.text("dataset MNIST de 28x28 pixels.")
