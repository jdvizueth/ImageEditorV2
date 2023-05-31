import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
import os
import math
import json
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from rembg import remove
import image_editorUI
import requests
from io import BytesIO

BUTTON_WIDTH = 1
SLIDER_LENGTH = 250

LEFT_IMAGE = 0
RIGHT_IMAGE = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AQUAMARINE = (212, 255, 127)

supportedFiletypes = [('JPEG Image', '*.jpg'), ('PNG Image', '*.png'),
                      ('PPM Image', '*.ppm')]


class BaseFrame(tk.Frame):
    '''The base frame inherited by all the tabs in the UI.'''

    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.root = root

        # UI elements which appear in all frames
        # We don't specify the position here, because it varies among
        # different frames
        self.status = tk.Label(self, text='Load an image to remove background')

        self.thresholdLabel = tk.Label(self, text='Threshold (10^x):')
        self.thresholdSlider = tk.Scale(self, from_=-4, to=0, resolution=0.1,
                                        orient=tk.HORIZONTAL, length=SLIDER_LENGTH)
        self.thresholdSlider.set(-2)

        self.imageCanvas = image_editorUI.ImageWidget(self)

        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        self.grid_rowconfigure(3, weight=1)

    def setStatus(self, text):
        self.status.configure(text=text)


class ImageGenerationFrame(BaseFrame):
    def __init__(self, parent, root):
        super().__init__(parent, root)

        self.screenshotButton = tk.Button(self, text='Save Image',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=5, sticky=tk.W+tk.E)

        self.generateButton = tk.Button(
            self, text='Generate Image', command=self.generateImage, width=BUTTON_WIDTH)
        self.generateButton.grid(
            row=0, column=0, sticky=tk.W+tk.E)

        self.textEntry = tk.Entry(self)  # Create a text entry widget
        # Adjust the row and column as needed
        self.textEntry.grid(rowspan=2, columnspan=6,
                            sticky=tk.N+tk.S+tk.E+tk.W)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None

    def get_text(self):
        return self.textEntry.get()

    def generateImage(self):
        # text = self.get_text()
        # if (text == ''):
        #     self.setStatus('Please write a description of the desired image')
        # else:
        #     self.setStatus('Generating Image...')
        #     r = requests.post("https://api.deepai.org/api/text2img",
        #                       data={
        #                           'text': text,
        #                       },
        #                       headers={
        #                           'api-key': 'b10bfb39-9d84-45e8-bc89-8f8f57c220e1',
        #                           'grid_size': "1"}
        #                       )
        #     print(r.json())
        #     response = r.json()
        #     if 'output_url' in response:
        #         image_url = response['output_url']
        #         image_data = requests.get(
        #             image_url).content  # Fetch the image data
        #         # Open the image using PIL
        #         image = Image.open(BytesIO(image_data))
        #         # # Convert the image to a Tkinter-compatible format
        #         # photo = ImageTk.PhotoImage(image)
        #         photo = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #         # self.imageCanvas.config(image=photo)  # Set the image in the label
        #         # # Keep a reference to prevent image garbage collection
        #         # self.imageCanvas.image = photo
        #         self.imageCanvas.drawCVImage(photo)
        #         self.setStatus('Image Generated Successfully')
        #     else:
        #         self.setStatus('Error Generating Image')
        self.setStatus('Sorry, our image generation is out of order :()')

    def screenshot(self):
        if self.image is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                                                      filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved image to ' + filename)
        else:
            image_editorUI.error('Load image before taking a screenshot!')


class ConvolveFrame(BaseFrame):
    def __init__(self, parent, root):
        super().__init__(parent, root)

        self.loadImageButton = tk.Button(self, text='Load Image',
                                         command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=5, column=0, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Save Image',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=5, column=5, sticky=tk.W+tk.E)

        self.convolveButton = tk.Button(
            self, text='Convolve', command=self.applyConvolution, width=BUTTON_WIDTH)

        self.convolveButton.grid(row=5, column=1, sticky=tk.W+tk.E)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None

        self.matrix_entries = []

        for i in range(3):
            row_entries = []
            for j in range(3):
                entry = tk.Entry(self)
                entry.grid(row=i, column=j+1, padx=5, pady=5)
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)

        self.entrymult = tk.Entry(self)
        self.entrymult.grid(row=1, column=0, padx=5, pady=5)

    def applyConvolution(self):
        # Retrieve the entered matrix values
        matrix = []
        for i in range(3):
            row_values = []
            for j in range(3):
                entry = self.matrix_entries[i][j]
                try:
                    value = float(entry.get())
                except ValueError:
                    value = 0.0
                row_values.append(value)
            matrix.append(row_values)

        # Perform the convolution with the entered matrix
        if self.image is not None:
            self.convolve(matrix)

    def convolve(self, matrix):

        kernel = np.array(matrix)
        kernel = float(self.entrymult.get()) * kernel
        print(kernel)

        convolved = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        self.image = np.copy(convolved)
        self.imageCanvas.drawCVImage(convolved)
        self.setStatus('Convolved')

    def loadImage(self):
        filename = tkFileDialog.askopenfilename(parent=self.root,
                                                filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
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
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                                                      filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved image to ' + filename)
        else:
            image_editorUI.error('Load image before taking a screenshot!')


class EditImageFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        self.cropping = False
        self.image = None
        self.image2 = None

        self.loadImageButton = tk.Button(self, text='Load Image',
                                         command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Save Image',
                                          command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=5, sticky=tk.W+tk.E)

        self.computeRemoverButton = tk.Button(
            self, text='Remove Background', command=self.computeRemove, width=BUTTON_WIDTH)

        self.computeRemoverButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.cropButton = tk.Button(
            self, text='Crop Image', command=self.croppingImage, width=BUTTON_WIDTH)

        self.cropButton.grid(row=0, column=2, sticky=tk.W+tk.E)

        self.rotateButton = tk.Button(
            self, text='Rotate Image', command=self.rotateImage, width=BUTTON_WIDTH)

        self.rotateButton.grid(row=0, column=3, sticky=tk.W+tk.E)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None

    def loadImage(self):
        filename = tkFileDialog.askopenfilename(parent=self.root,
                                                filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
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
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
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

    def crop_image(self, image, start_x, start_y, end_x, end_y):
        cropped_image = image[start_y:end_y, start_x:end_x]
        return cropped_image

    def mouse_callback(self, event, x, y, flags, param):
        # Access the global variables

        if event == cv2.EVENT_LBUTTONDOWN:
            # Initialize the starting coordinates
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y
            self.image = np.copy(self.image2)
            cv2.imshow('Press q to quit, c to crop', self.image)
            self.cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            # Update the ending coordinates and indicate that cropping is finished
            self.end_x, self.end_y = x, y
            self.cropping = False

            # Ensure the coordinates are valid (start < end)
            self.start_x, self.start_y = min(
                self.start_x, self.end_x), min(self.start_y, self.end_y)
            self.end_x, self.end_y = max(
                self.start_x, self.end_x), max(self.start_y, self.end_y)

            self.image = np.copy(self.image2)

            cv2.rectangle(self.image, (self.start_x, self.start_y),
                          (self.end_x, self.end_y), (0, 255, 0), 2)

            cv2.imshow('Press q to quit, c to crop', self.image)

    def croppingImage(self):
        if self.image is not None:
            self.image2 = np.copy(self.image)
            cv2.namedWindow('Press q to quit, c to crop')
            cv2.setMouseCallback(
                'Press q to quit, c to crop', self.mouse_callback)

            while True:
                cv2.imshow('Press q to quit, c to crop', self.image)

                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    self.image = np.copy(self.image2)
                    break

                if key == ord('c') and not self.cropping:
                    cropped_image = self.crop_image(
                        self.image2, self.start_x, self.start_y, self.end_x, self.end_y)
                    # self.image = np.copy(self.image2)
                    # cv2.imshow('Cropped Image', cropped_image)
                    self.image = np.copy(cropped_image)
                    self.imageCanvas.drawCVImage(cropped_image)
                    # cv2.waitKey(0)
                    break

            cv2.destroyAllWindows()
        else:
            image_editorUI.error('Load image before cropping!')

    def rotateImage(self):
        if self.image is not None:
            rotated = np.copy(cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE))
            self.image = np.copy(rotated)
            self.imageCanvas.drawCVImage(rotated)
        else:
            image_editorUI.error('Load image before rotating')


class ImageEditorFrame(tk.Frame):
    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = root
        self.notebook = ttk.Notebook(self.parent)
        # Add Sections and Buttons here that specifies what to do
        self.ImageEditingFrame = EditImageFrame(
            self.notebook, root)

        self.notebook.add(self.ImageEditingFrame,
                          text='Edit Pictures Tab')

        self.ConvolveFrame = ConvolveFrame(self.notebook, root)

        self.notebook.add(self.ConvolveFrame, text="Colvolve an Image")

        self.ImageGenerationFrame = ImageGenerationFrame(self.notebook, root)

        self.notebook.add(self.ImageGenerationFrame, text='Generate an Image')

        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)

    def CloseWindow(self):
        self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = ImageEditorFrame(root, root)
    root.title('Image Editor')
    # Put the window on top of the other windows
    # root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry("%dx%d+0+0" % (w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    # root.wm_attributes('-topmost', 1)
    root.mainloop()
