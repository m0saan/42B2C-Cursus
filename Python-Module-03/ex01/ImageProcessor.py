import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    A class that provides methods for loading and displaying images.

    Methods
    -------
    load(path: str) -> np.ndarray:
        Opens the PNG file specified by the path argument and returns an
        array with the RGB values of the pixels image. It also prints a message
        specifying the dimensions of the image (e.g. 340 x 500).

        Parameters
        ----------
        path : str
            The path to the image file.

        Returns
        -------
        np.ndarray
            The RGB values of the pixels image as a numpy array.

        Raises
        ------
        FileNotFoundError
            If the file passed as argument does not exist.
        Exception
            If the file passed as argument can't be read as an image.

    display(array: np.ndarray) -> None:
        Takes a numpy array as an argument and displays the corresponding RGB image.

        Parameters
        ----------
        array : np.ndarray
            A numpy array with RGB values of pixels.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the numpy array passed as argument can't be displayed as an image.
    """
    
    def load(self, path: str) -> np.ndarray:
        
        """
        Opens the PNG file specified by the path argument and returns an
        array with the RGB values of the pixels image. It also prints a message
        specifying the dimensions of the image (e.g. 340 x 500).

        Parameters
        ----------
        path : str
            The path to the image file.

        Returns
        -------
        np.ndarray
            The RGB values of the pixels image as a numpy array.

        Raises
        ------
        FileNotFoundError
            If the file passed as argument does not exist.
        Exception
            If the file passed as argument can't be read as an image.
        """
        
        try:
            with Image.open(path) as im_file:
                img_arr = np.array(im_file)
                print(f'Image dimension: {img_arr.shape[0]}, {img_arr.shape[1]}')
                return img_arr
        except FileNotFoundError:
            print(f"Error: File {path} not found.")
        except:
            print(f"Error: Could not read {path} as an image.")
        
    def display(self, array: np.ndarray) -> None:
        """
        Takes a numpy array as an argument and displays the corresponding RGB image.

        Parameters
        ----------
        array : np.ndarray
            A numpy array with RGB values of pixels.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the numpy array passed as argument can't be displayed as an image.
        """
        try:
            plt.imshow(array)
            plt.axis('off')
            plt.show()
        except:
            print('Error: could not display the image')

    
    
if __name__ == '__main__':
    imp = ImageProcessor()
    arr = imp.load("non_existing_file.png")
    
    arr = imp.load("empty_file.png")
    print(arr)
    
    arr = imp.load("../resources/42AI.png")
    print(arr)