import numpy as np

class ScrapBooker:
    
    def crop(self, array: np.ndarray, dim: tuple, position: tuple=(0, 0)):
        """
        Crops the image as a rectangle via dim arguments (being the new height
        and width of the image) from the coordinates given by position arguments.
        
        Args:
        -----
        array: numpy.ndarray
        dim: tuple of 2 integers.
        position: tuple of 2 integers.
        
        Return:
        -------
        new_arr: the cropped numpy.ndarray.
        None (if combination of parameters not compatible).
        
        Raise:
        ------
        This function should not raise any Exception.
        """
        h, w = array.shape[:2]
        x,y = position
        x_new, y_new = dim
        
        if x + x_new > h or y + y_new > w or x < 0 or y < 0 or x_new <= 0 or y_new <= 0:
            print("Error: Invalid parameters for cropping."); return None
        
        cropped = array[x:x+x_new, y:y+y_new]
        return cropped
        
    
    def thin(self, array: np.ndarray, n: int, axis: int):
        """
        Deletes every n-th line pixels along the specified axis (0: Horizontal, 1: Vertical).
        
        Args:
        -----
        array: numpy.ndarray.
        n: non null positive integer lower than the number of row/column of the array
           (depending on axis value).
        axis: positive non null integer.
        
        Return:
        -------
        new_arr: thinned numpy.ndarray.
        None (if combination of parameters not compatible).
        
        Raise:
        ------
        This function should not raise any Exception.
        """
        
        if n <= 0 or (axis != 0 and axis != 1):
            print("Error: invalid input parameters.")
            return None

        if axis == 1:
            # Delete every n-th row
            rows_to_delete = np.arange(start=n - 1, stop=array.shape[0],  step=n)
            new_array = np.delete(array, rows_to_delete, axis=0)
        else:
            # Delete every n-th column
            cols_to_delete = np.arange(n - 1, array.shape[1], n)
            print(cols_to_delete)
            new_array = np.delete(array, cols_to_delete, axis=1)

        return new_array
    

        
    
    def juxtapose(self, array: np.ndarray, n: int, axis: int):
        """
        Juxtaposes n copies of the image along the specified axis.
        
        Args:
        -----
        array: numpy.ndarray.
        n: positive non null integer.
        axis: integer of value 0 or 1.
        
        Return:
        -------
        new_arr: juxtaposed numpy.ndarray.
        None (if combination of parameters not compatible).
        
        Raises:
        -------
        This function should not raise any Exception.
        """
        
        if n < 1 or axis not in (0, 1):
            return None
        new_arr =  np.concatenate([array for _ in range(n)], axis=axis)
        return new_arr
    
    def mosaic(self, array: np.ndarray, dim: tuple):
        """
        Makes a grid with multiple copies of the array. The dim argument specifies
        the number of repetition along each dimensions.
        
        Args:
        -----
        array: numpy.ndarray.
        dim: tuple of 2 integers.
        
        Return:
        -------
        new_arr: mosaic numpy.ndarray.
        None (if combination of parameters not compatible).
        
        Raises:
        -------
        This function should not raise any Exception.
        """
        
        return np.array(np.tile(array, dim))

if __name__ == '__main__':\
    
    spb = ScrapBooker()
    # arr1 = np.arange(0,25).reshape(5,5)
    # cropped = spb.crop(arr1, (3,1),(1,0))
    # print(cropped)
    # print(cropped.shape)
    # Output :
    # array([[ 5],
    # [10],
    # [15]])
    
    
    # arr2 = np.array("A B C D E F G H I".split() * 6).reshape(-1,9)
    # print(arr2)
    # print('\n--------------------------------------------------\n')
    # print(spb.thin(arr2,2,1))
    
    arr3 = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
    print(spb.juxtapose(arr3, 3, 0))
    
    # array([[1, 2, 3, 1, 2, 3, 1, 2, 3],
    # [1, 2, 3, 1, 2, 3, 1, 2, 3],
    # [1, 2, 3, 1, 2, 3, 1, 2, 3]]) 1 2