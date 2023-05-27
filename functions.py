import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from rembg import remove
import image_editorUI
import frames

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AQUAMARINE = (212, 255, 127)

supportedFiletypes = [('JPEG Image', '*.jpg'), ('PNG Image', '*.png'),
                      ('PPM Image', '*.ppm')]


def loadImage(self):
    filename = image_editorUI.tkFileDialog.askopenfilename(parent=self.root,
                                                           filetypes=supportedFiletypes)
    if filename and frames.os.path.isfile(filename):
        self.image = cv2.imread(filename)
        self.imageCanvas.drawCVImage(self.image)
        self.setStatus('Loaded ' + filename)


def reloadImage(self, image):
    if self.image is not None:
        self.keypoints = None
        self.image = image
        self.imageCanvas.drawCVImage(self.image)


def screenshot(self):
    if self.image is not None:
        filename = image_editorUI.tkFileDialog.asksaveasfilename(parent=self.root,
                                                                 filetypes=supportedFiletypes, defaultextension=".png")
        if filename:
            self.imageCanvas.writeToFile(filename)
            self.setStatus('Saved image to ' + filename)
    else:
        image_editorUI.error('Load image before taking a screenshot!')


def computeRemove(self, *args):
    if self.image is not None:
        self.setStatus('Clearing Background')

        output = remove(self.image)

        self.reloadImage(output)

        self.setStatus('Cleared Background')
