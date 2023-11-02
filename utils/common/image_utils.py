import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import PIL.Image
from shapely.geometry import Polygon
from pathlib import Path
import base64

class Image:
    @staticmethod
    def show(image, max_res=768):
        cv2.imshow('image', Image.resize_max(image, max_res))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    @staticmethod
    def show_pillow(image):
        PIL.Image.fromarray(image).show()

    @staticmethod
    def save(image, filename):
        if Path(filename).parents[0].exists() == False:
            Path(filename).parents[0].mkdir(parents=True, exist_ok=True) 
        cv2.imwrite(filename, image)

    @staticmethod
    def resize(image, fxy):
        return cv2.resize(image, (0, 0), fx=fxy, fy=fxy)
    
    @staticmethod
    def is_portrait(shape, threshold=1):
        return shape[0] * threshold > shape[1]

    @staticmethod
    def resize_max(image, max_res):
        max_dim = max(image.shape)
        if max_dim > max_res:
            resize_scale = max_res / max_dim
        else:
            max_dim_arg = np.argmax(image.shape)
            resize_scale = max_res / image.shape[max_dim_arg]
        image = cv2.resize(image, (0, 0), fx=resize_scale, fy=resize_scale)
        return image

    @staticmethod
    def rotate(image, angle, r90cw=False):
        if r90cw:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    @staticmethod
    def base64_to_numpy(base64_string):
        imgdata = base64.b64decode(base64_string)
        return cv2.imdecode(np.frombuffer(imgdata, np.uint8), cv2.IMREAD_COLOR)
    
    @staticmethod
    def numpy_to_base64(image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def augment(image, angle=[0, 360], r90cw=[0, 1], flip=[0, 1]):
        image = Image.rotate(
            image, 
            np.random.randint(*angle), 
            r90cw=bool(np.random.randint(*r90cw))
        )
        image = np.fliplr(image) if np.random.randint(*flip) else image
        image = np.flipud(image) if np.random.randint(*flip) else image
        return image

class Intersection:
    @staticmethod
    def is_inside(xyxyA, xyxyB, threshold=0.5):
        min_area, max_area = sorted([
        	Intersection.reactangle_area(xyxyA),
            Intersection.reactangle_area(xyxyB)])
        overlap_area = Intersection.overlap_area(xyxyA, xyxyB)
        if (overlap_area/min_area) >= threshold:
            return True
        return False

    @staticmethod
    def group_subsets(sets, subsets, threshold):
        grouped_sets = {}
        for set_ in sets:
            grouped_sets[set_] = []
            for subset in subsets:
                if Intersection.is_inside(set_, subset, threshold):
                    grouped_sets[set_].append(subset)
        return grouped_sets

    @staticmethod
    def reactangle_area(xyxy):
        return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
    
    @staticmethod
    def overlap_area(xyxyA, xyxyB):
        coordsA = [(xyxyA[0], xyxyA[1]), (xyxyA[2], xyxyA[1]), (xyxyA[2], xyxyA[3]), (xyxyA[0], xyxyA[3])]
        coordsB = [(xyxyB[0], xyxyB[1]), (xyxyB[2], xyxyB[1]), (xyxyB[2], xyxyB[3]), (xyxyB[0], xyxyB[3])]
        polyA = Polygon(coordsA)
        polyB = Polygon(coordsB)
        return polyA.intersection(polyB).area


    @staticmethod    
    def union_area(xyxyA, xyxyB):
        areaA = Intersection.reactangle_area(xyxyA)
        areaB = Intersection.reactangle_area(xyxyB)
        return areaA + areaB - Intersection.overlap_area(xyxyA, xyxyB)
    
    @staticmethod
    def intersection_over_union(xyxyA, xyxyB, threshold):
        aoo = Intersection.overlap_area(xyxyA, xyxyB)
        aou = Intersection.union_area(xyxyA, xyxyB)
        iou = aoo / aou
        return iou >= threshold

if __name__ == '__main__':
    from glob import glob
    im = Image()
    for i in glob(r'F:\sevencred\leitorDocumentos\data\raw_images\*'):
        image = cv2.imread(i)
        im.show_pillow(im.filter_ocr(image))
        input()