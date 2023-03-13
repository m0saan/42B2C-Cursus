import numpy as np
from ImageProcessor import ImageProcessor


class ColorFilter:
    
    def invert(self, array: np.ndarray):
        """
        Inverts the color of the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        max_value = 255
        # we have an extra alpha channel that we don't want to modify, or the
        # transparent parts of the image won't be transparent anymore.
        # We can use numpy slice notation to modify all dimensions of the array.
        # the [:, :, :3] indexing syntax is used to select the first three dimensions
        # of the input array, corresponding to the color channels.
        print(array[:, :, :3].shape)
        return max_value - array[:, :, :3]
    
    def to_blue(self, array):
        """
        Applies a blue filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        
        # set the RED and BLUE channels to zero.
        new_arr = array.copy()
        new_arr[:, :, 0] = 0
        new_arr[:, :, 1] = 0
        
        return new_arr
    
    def to_green(self, array):
        """
        Applies a green filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        
        # set the RED and BLUE channels to zero
        new_arr = array.copy()
        new_arr[:,:,0] = 0
        new_arr[:,:,2] = 0
        return new_arr
    
    def to_red(self, array):
        """
        Applies a red filter to the image received as a numpy array.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        # set the GREEN and BLUE channels to zero
        new_arr = array.copy()
        new_arr[:,:,1] = 0
        new_arr[:,:,2] = 0
        return new_arr
    
    def to_celluloid(self, array):
        """
        Applies a celluloid filter to the image received as a numpy array.
        Celluloid filter must display at least four thresholds of shades.
        Be careful! You are not asked to apply black contour on the object,
        you only have to work on the shades of your images.
        Remarks:
        celluloid filter is also known as cel-shading or toon-shading.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        raise NotImplementedError()
        
    def to_grayscale(self, array, filter, **kwargs):
        """
        Applies a grayscale filter to the image received as a numpy array.
        For filter = ’mean’/’m’: performs the mean of RBG channels.
        For filter = ’weight’/’w’: performs a weighted mean of RBG channels.
        Args:
        -----
        array: numpy.ndarray corresponding to the image.
        filter: string with accepted values in [’m’,’mean’,’w’,’weight’]
        weights: [kwargs] list of 3 floats where the sum equals to 1,
        corresponding to the weights of each RBG channels.
        Return:
        -------
        array: numpy.ndarray corresponding to the transformed image.
        None: otherwise.
        Raises:
        -------
        This function should not raise any Exception.
        """
        
        raise NotImplementedError()
        
    
if __name__ == '__main__':
    imp = ImageProcessor()
    arr = imp.load("elon_canaGAN.png")
    
    cf = ColorFilter()
    inverted = cf.invert(arr)
    imp.display(inverted)
    
    # imp.display(cf.to_blue(arr))
    # imp.display(cf.to_green(arr))
    # imp.display(cf.to_red(arr))
    imp.display(cf.to_grayscale(arr, "m"))
    
    
    
    