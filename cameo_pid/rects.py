import cv2
import numpy
import utils


def outlineRect(image, rect, color):
    """Dibuja un rectángulo sobre una imagen"""
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color)

def copyRect(src, dst, srcRect, dstRect, mask = None,
             interpolation = cv2.INTER_LINEAR):
    """Copia una región rectangular en otra región """

    x0, y0, w0, h0 = srcRect    # Región origen
    x1, y1, w1, h1 = dstRect    # Región destino

    # Recorta la región de origen, la redimensiona al tamaño de destino y la pega en destino.
    if mask is None:
        dst[y1:y1+h1, x1:x1+w1] = \
            cv2.resize(src[y0:y0+h0, x0:x0+w0], (w1, h1),
                       interpolation = interpolation)
    else:
        if not utils.isGray(src):
            # Si la imagen es a color, convierte la máscara a tres canales.
            mask = mask.repeat(3).reshape(h0, w0, 3)
        # Redimensiona máscara y fragmento origen
        dst[y1:y1+h1, x1:x1+w1] = \
            numpy.where(cv2.resize(mask, (w1, h1),
                                   interpolation = \
                                   cv2.INTER_NEAREST),
                        cv2.resize(src[y0:y0+h0, x0:x0+w0], (w1, h1),
                                   interpolation = interpolation),
                        dst[y1:y1+h1, x1:x1+w1])


def swapRects(src, dst, rects, masks = None,
              interpolation = cv2.INTER_LINEAR):
    """Permite intercambiar varias regiones rectangulares dentro de una imagen de forma circular"""

    if dst is not src:  # Si destino es diferente de origen copia la región origen en destino.
        dst[:] = src

    numRects = len(rects)
    if numRects < 2:    # Si hay menos de dos regiones no hay nada para intercambiar
        return

    if masks is None:
        masks = [None] * numRects

    # Guardar copia de la última región
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y+h, x:x+w].copy()

    # Mover contenido del rectángulo i al i+1
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i+1], masks[i],
                 interpolation)
        i -= 1

    # Copiar contenido del último rectángulo en el primero
    copyRect(temp, dst, (0, 0, w, h), rects[0], masks[numRects - 1],
             interpolation)
