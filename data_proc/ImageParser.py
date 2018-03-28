from PIL import Image

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    # cropped_image.show()


def crop_resize(image_path, coords, saved_location, size=(32,32)):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    resized_image = cropped_image.resize(size, resample=0)
    resized_image.save(saved_location)
    # cropped_image.show()


def get_crop_resize_virtualize(image_path, coords, mirror, rot_degree = 0, size=(100, 100)):
    """
    Return cropped,rotated image from path in specified resolution and
    apply virtualization transformations.
    :param image_path: path to image
    :param coords: coordination of cropping x1,y1,x2,y2
    :param size: (touple) final resolution, default is 100x100 pixels
    :param rot_degree: (int) degrees to rotate
    :return:
    """
    image_obj = Image.open(image_path)
    #check coordinates
    #need to create cords, because in case of evaluation they are returned as tuple from dictionary
    cords = [0 for _ in range(4)]
    cords[0] = max(0, coords[0])
    cords[1] = max(0, coords[1])
    cords[2] = min(image_obj.size[0], coords[2])
    cords[3] = min(image_obj.size[1], coords[3])
    if mirror:
        image_obj = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    if rot_degree != 0:
        # Crop -> Rotate -> Resize -> Return
        # for more option on rotation see http://matthiaseisen.com/pp/patterns/p0201/
        # name = "rot_" + str(rot_degree) + "_mir_" + str(mirror) + ".jpg"
        # image_obj.crop(coords).rotate(rot_degree, expand=False).resize(size, resample=Image.BILINEAR).save(name)
        return image_obj.crop(cords).rotate(rot_degree, expand=False).resize(size, resample=Image.BILINEAR)
    else:
        # Crop -> Resize -> Return
        # name = "rot_" + str(rot_degree) + "_mir_" + str(mirror) + ".jpg"
        # image_obj.crop(coords).rotate(rot_degree, expand=False).resize(size, resample=Image.BILINEAR).save(name)
        return image_obj.crop(cords).resize(size, resample=Image.BILINEAR)


def get_crop_resize(image_path, coords, size=(100, 100)):
    """
    Return cropped,rotated image from path in specified resolution.
    :param image_path: path to image
    :param coords: coordination of cropping x1,y1,x2,y2
    :param size: (touple) final resolution, default is 100x100 pixels
    :return:
    """
    image_obj = Image.open(image_path)
    #check coordinates
    #need to create cords, because in case of evaluation they are returned as tuple from dictionary
    cords = [0 for _ in range(4)]
    cords[0] = max(0, coords[0])
    cords[1] = max(0, coords[1])
    cords[2] = min(image_obj.size[0], coords[2])
    cords[3] = min(image_obj.size[1], coords[3])
    return image_obj.crop(cords).resize(size, resample=Image.BILINEAR)
