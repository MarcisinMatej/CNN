from data_proc import ImageParser

PATH = 'data_proc/config_files/'


def load_crop_txts():
    with open(PATH+'bboxes.txt') as file_bboxes:
        bboxes = file_bboxes.readlines()
    with open(PATH+'folder.txt') as file_folder:
        folder = file_folder.readlines()
    return bboxes, folder


def crop_store_images(bboxes, folders, size=(64, 64)):
    images = []
    for bbox, folder in zip(bboxes,folders):
        arr = bbox.split()
        img = {
            'path': arr[0],
            'bbox': (float(arr[1]), float(arr[2]), float(arr[5]), float(arr[6])),
            'name': arr[0].split('/')[-1]
        }
        code = int(folder.split()[-1])
        # parse output folder
        if code == 1:
            img.update({'folder':'data/train/'})
        elif code == 2:
            img.update({'folder':'data/test/'})
        if code == 3:
            img.update({'folder':'data/validation/'})
        images.append(img)
    for img in images:
        # TODO change here we crop and resize
        ImageParser.crop_resize(img['path'], img['bbox'], img['folder'] + img['name'], size=size)


def run_data_crop(size):
    '''
    method processes images in such a manner, that it will cut out face and store it in separate folders (training,testing,validaiton)
    :return:
    '''
    bboxes, folder = load_crop_txts()
    crop_store_images(bboxes, folder,size=size)
    # CelebA/img_align_celeba/000001.jpg 45 76 148 76 148 179 45 179


if __name__ == "__main__":
    run_data_crop((64, 64))