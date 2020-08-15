# VIA_Image_Split

The use case is when your image is too large with many small annotations.

Matterport's Mask RCNN rescales an input image to 1024 x 1024 and only considers top-100 objects, and if your image is too big and objects are too small, it cannot train well after rescaling as objects can become one pixel or only a couple of pixels.

A solution is to split big images into smaller pieces.

What this code does is to take a folder of images along with an VIA annotation file as input, and outputs a folder of processed images and an annotation file for them, where each output image is obtained by a sliding window on an input image.

The sliding window has parameter (width, height, overlap), where overlap defines how many pixels are overlapped with the previous window along each side during sliding, so that we can guarantee that a boundary object that is not complete in one window will be fully contained in another window.

Try out .ipynb file on our input data under folder "test" to understand how it use it :-)

Contributors:
* Robertson, John P
* Yan, Da (yanda@uab.edu)
