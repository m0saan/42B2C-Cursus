# README

## XXX

we recommend that you choose a language with a library that facilitates image-processing workflow like plantCV
Different methods of direct extraction of characteristics from an image of a leaf need
to be implemented. Once again, you must display at least 6 image transformations.

For exemple:
    $> ./Transformation.[extension] image.jpg

If your program is given a direct path to an image, it must display your set of image
transformations. However, if it is given a source path to a directory filled with multiple
images, it must then save all the image transformations in the specified destination directory.

For exemple:
    $> ./Transformation.[extension] -src Apple/apple_healthy/ -dst dst_directory -mask

Think to make your own usage to facilitate the choice of the
arguments with ./Transformation.[extension] -h

To allow a fast evaluation, it goes without saying that you will
create your own data set from the images by following all these steps
beforehand. Beware, the evaluation will assess whether your program
is working well on small data sets.

### Commandes

    - pyinstaller --onefile Transformation.py

    - python3 -m venv ./venv

## KEYWORDS

Creating a mask from an image usually involves isolating certain parts of the image based on some criteria, like color ranges, brightness, or edges. The mask can then be used for various image processing tasks, such as background removal, object isolation, or change detection.

Here’s a step-by-step explanation of how you might transform an original image into a mask using Python, specifically with the help of libraries such as NumPy and OpenCV:

Read the Image: Load the original image using OpenCV.
python
Copy code
import cv2
original_image = cv2.imread('path_to_image.jpg')
Preprocess the Image: Depending on the requirements, you might need to convert the image to grayscale or apply some filters to enhance the features you want to isolate.
python
Copy code
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
or for color-based isolation:

python
Copy code
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
Thresholding: Apply a threshold to create a binary image where the pixels of interest are white (255) and all other pixels are black (0). This can be done using a simple threshold, adaptive threshold, or even color-based thresholding.
python
Copy code
_, binary_mask = cv2.threshold(grayscale_image, thresh, 255, cv2.THRESH_BINARY)
or for color thresholding:

python
Copy code
lower_bound = np.array([h_min, s_min, v_min]) # define lower bound of color
upper_bound = np.array([h_max, s_max, v_max]) # define upper bound of color
color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
Refine the Mask: Use morphological operations such as erosion and dilation to remove noise and fill gaps in the mask.
python
Copy code
refined_mask = cv2.erode(binary_mask, None, iterations=2)
refined_mask = cv2.dilate(refined_mask, None, iterations=2)
Apply the Mask: Use the mask to isolate the object of interest from the original image.
python
Copy code
result = cv2.bitwise_and(original_image, original_image, mask=refined_mask)
Save or Display the Mask and Result: Finally, you can save or display the mask and the resulting image.
python
Copy code
cv2.imwrite('mask.jpg', refined_mask)
cv2.imwrite('result.jpg', result)
cv2.imshow('Mask', refined_mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
The threshold values and the color bounds need to be chosen based on the specific characteristics of the object you are trying to isolate. These values can be found through experimentation or using techniques like histogram analysis.

Would you like a sample code that you can run with an actual image to see this process in action?





