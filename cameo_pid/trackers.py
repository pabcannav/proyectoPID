import cv2
import rects
import utils


class Face(object):
    """Datos sobre rasgos faciales."""

    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None

class FaceTracker(object):
    """Rastreador de rasgos faciales."""

    def __init__(self, scaleFactor = 1.2, minNeighbors = 2,
                 flags = cv2.CASCADE_SCALE_IMAGE):

        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []

        self._faceClassifier = cv2.CascadeClassifier(
            'cameo_pid\cascades\haarcascade_frontalface_alt.xml')   # Clasificador de Haar preentrenado para detectar rostros frontales.
        self._eyeClassifier = cv2.CascadeClassifier(
            'cameo_pid\cascades\haarcascade_eye.xml')   # Clasificador de Haar preentrenado para detectar ojos.

    @property
    def faces(self):
        """Los rasgos faciales rastreados."""
        return self._faces

    def update(self, image):
        """Procesa una imagen para detectar rostros y ojos, y actualiza la lista interna de objetos Face."""

        self._faces = []

        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)

        minSize = utils.widthHeightDividedBy(image, 8)  # Tamaño mínimo para detectar (Evitamos detectar objetos demasiado pequeños)

        # A continuación ejecutamos el clasificador para detectar rostros en la imagen. Devuelve una lista de rectángulos
        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags,
            minSize)

        # Si se detecta algún rostro.
        if faceRects is not None:
            for faceRect in faceRects:  # Se recorre cada rostro

                # Crear nueva instancia de face y asignar rectángulo facial
                face = Face()
                face.faceRect = faceRect

                x, y, w, h = faceRect   # Descomponer coordenadas del rectángulo del rostro.

                # Busca el ojo izquierdo dentro del rectángulo
                searchRect = (x+w//7, y, w*2//7, h//2)
                face.leftEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                # Busca el ojo derecho dentro del rectángulo
                searchRect = (x+w*4//7, y, w*2//7, h//2)
                face.rightEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                self._faces.append(face)    # Añade el rostro a la lista de rostros detectados.

    def _detectOneObject(self, classifier, image, rect,
                          imageSizeToMinSizeRatio):

        x, y, w, h = rect   # Extraer coordenadas y dimensión del rectángulo de búsqueda

        minSize = utils.widthHeightDividedBy(
            image, imageSizeToMinSizeRatio) # Tamaño mínimo de los objetos detectados

        subImage = image[y:y+h, x:x+w]  # Obtener subimagen donde se realizará la detección

        # Usar clasificador haar para buscar objetos dentro de la subimagen
        subRects = classifier.detectMultiScale(
            subImage, self.scaleFactor, self.minNeighbors,
            self.flags, minSize)

        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]    # Tomar primer objeto detectado
        return (x+subX, y+subY, subW, subH)     # Convertir coordenadas relativas a coordenadas obsolutas

    # Dibujar rectangulos alrededor del rostro
    def drawDebugRects(self, image):

        # Si la imagen es gris utiliza el color blanco
        if utils.isGray(image):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
        else:   # Si la imagen es a color asigno los siguientes colores
            faceColor = (255, 255, 255) # blanco para el rectángulo de la cara
            leftEyeColor = (0, 0, 255) # rojo para el rectángulo del ojo izquierdo
            rightEyeColor = (0, 255, 255) # amarillo para el rectangulo del ojo derecho
        
        # Dibujo los rectangulos alrededor de la cara y de los ojos
        for face in self.faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect,
                              rightEyeColor)
