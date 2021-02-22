import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import glob
import os



def resize_square(img):
    """長辺のサイズで正方形の画像に"""
    l=max(img.shape[:2])
    
    h,w = img.shape[:2]
    hm = (l-h)//2
    wm = (l-w)//2
    return cv2.copyMakeBorder(img,
                            hm,
                            hm+(l-h)%2,
                            wm,
                            wm+(l-w)%2,
                            cv2.BORDER_CONSTANT,
                            value=0)

class CropLemon(ImageOnlyTransform):
    """レモンが写っている部分をcrop"""

    def __init__(self, margin=10, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.margin = margin

    def get_box(self, img):
        """ 中央に近い黄色い領域を見つける """
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        # h,v のしきい値で crop
        _, img_hcrop = cv2.threshold(h, 0, 40, cv2.THRESH_BINARY)
        _, img_vcrop = cv2.threshold(v, v.mean(), 255, cv2.THRESH_BINARY)
        th_img = (img_hcrop * (img_vcrop / 255)).astype(np.uint8)

        contours, hierarchy = \
            cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # サイズの大きいものだけ選択
        contours = [c for c in contours if cv2.contourArea(c) > 10000]
        if not contours: return None

        # 中央に近いものを選択
        center = np.array([img.shape[1] / 2, img.shape[0] / 2])  # w, h
        min_contour = None
        min_dist = 1e10

        for c in contours:
            tmp = np.array(c).reshape(-1, 2)
            m = tmp.mean(axis=0)
            dist = sum((center - m) ** 2)
            if dist < min_dist:
                min_contour = tmp
                min_dist = dist

        box = [
            *(min_contour.min(axis=0) - self.margin).astype(np.int).tolist(),
            *(min_contour.max(axis=0) + self.margin).astype(np.int).tolist()]
        for i in range(4):
            if box[i] < 0: box[i] = 0
            if i % 2 == 0:
                if box[i] > img.shape[1]: box[i] = img.shape[1]
            else:
                if box[i] > img.shape[0]: box[i] = img.shape[0]

        return box  # left, top, right, bottom

    def apply(self, image, **params):
        image = image.copy()
        box = self.get_box(image)
        crop_img = None
        if not box or (box[3] - box[1] < 50 or box[2] - box[0] < 50):
            pass
        else:
            try:
                crop_img = image[box[1]:box[3], box[0]:box[2]]
            except:
                pass
        if crop_img is None:
            crop_img = image[40:, 10:-20]
        return resize_square(crop_img)

    def get_transform_init_args_names(self):
        return ("margin",)


img_dir = '/datas'
train_test_dir = ['/train_images','/test_images']
transform_lemons = '/transform_lemons'
place_lemon = os.path.dirname(__file__).rstrip('/src')


def make_dataset():
    for i in train_test_dir:
        files = glob.glob(place_lemon + img_dir + i +'/*.jpg')
        if os.path.isdir(place_lemon + img_dir + transform_lemons + i) is False:
            print('hello')
            os.makedirs(place_lemon + img_dir + transform_lemons + i)


        for j in files:
            img = cv2.imread(j)
            crop_img = CropLemon()(image=img)['image']
            index = '/' + j[-8:]
            
            cv2.imwrite(place_lemon + img_dir + transform_lemons + i + index,crop_img)
            

if __name__ == '__main__':
    
    files = glob.glob(place_lemon + '/datas/train_images/*.jpg')
    img = cv2.imread(files[0])
    print(len(files))
    if os.path.isdir(place_lemon + img_dir + transform_lemons + '/train_images') is False:
        os.makedirs(place_lemon + img_dir + transform_lemons + '/train_images')


    #img = cv2.imread(place_lemon+'/datas/train_images/')
    cv2.imwrite(place_lemon+"/src/test_img.png",img)
    """
    for i in train_test_dir:
        for j in files:
            img = cv2.imread(place_lemon + img_dir + i + j)
            cv2.imwrite("test_img",img)
    """

    """
    for i in train_test_dir:
        files = glob.glob(img_dir + i +'/*.jpg')
        print(len(files))
        for j in files:
            img = cv2.imread(img_dir + i + j)
            crop_img = CropLemon()(image=img)['image']
            cv2.imwrite("altY.png",crop_img)
            break
    """
