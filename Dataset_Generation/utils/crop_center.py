'''
This function is used to center-crop an image
INPUT:
- img: image to be cropped
- cropx: x-length of output image
- cropy: y-length of output image
'''
def crop_center(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    out = img[starty:starty+cropy,startx:startx+cropx, :] if len(img.shape)==3 else img[starty:starty+cropy,startx:startx+cropx]
    return out
