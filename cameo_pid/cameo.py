import cv2
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker


#INTERCAMBIO DE CARAS------------------------------------------------------------------------------------------
class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                            self.onKeypress)    # Gestionar ventana de visualización.
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)     # Usar cámara predeterminada
        self._faceTracker = FaceTracker()   # Usar FaceTraker. Se encarga de detectar caras en la imagen.
        self._shouldDrawDebugRects = False
        

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()  # Crear y mostrar ventana para visualizar el video.
        # El siguiente bucle se mantiene activo mientras la ventana está abierta.
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame  # Obtiene el fotograma actual.

            if frame is not None:

                self._faceTracker.update(frame) # Actualiza el rastreador de caras para el fotograma seleccionado
                faces = self._faceTracker.faces # Obtiene la lista de las caras detectadas.
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces])  # Intercambia las imágenes dentro de los rectangulos alrededor de las caras.

                

                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame) # Dibuja rectangulos alrededor de las caras detectadas.

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.

        space  -> Tomar captura de pantalla.
        tab    -> Activar/Desactivar grabación de pantalla.
        x      -> Activa/Desactivar rectangulo alrededor de las caras.
        escape -> Salir.

        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()
#-------------------------------------------------------------------------------------------------------------------------


if __name__=="__main__":
    Cameo().run() 
    