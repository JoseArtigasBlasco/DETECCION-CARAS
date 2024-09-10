import cv2
import os

# Cargar modelo Haar preentrenado para detección de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen {image_path}")
        return

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen con parámetros ajustados
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=7, minSize=(30, 30))

    # Dibujar rectángulos alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar la imagen con las caras detectadas
    resized_img = cv2.resize(image, (700, 500))
    cv2.imshow('Detected Faces', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la carpeta con las imágenes a analizar
folder_path = 'imagenes'

# Iterar sobre los archivos de la carpeta
for filename in os.listdir(folder_path):
    # Construir la ruta completa del archivo
    image_path = os.path.join(folder_path, filename)
    # Verificar que sea un archivo de imagen válido
    if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        detect_faces(image_path)


