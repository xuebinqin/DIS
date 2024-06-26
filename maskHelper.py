import cv2
# from u2net_helper import u2netHelper
import warnings
import numpy as np


class maskImage:
    def __init__(self):
        self.image = None
        self.mask = None
        self.left = None
        self.right = None
        self.points = None
        self.annotations = None
        self.masked_image = None
        
    
    def calculate_area(self,shape):
        area = 0
        n = len(shape)
        for i in range(n):
            x1, y1 = shape[i]
            x2, y2 = shape[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        return abs(area) / 2
        
        
    def maskProcessor(self,mask, tolerance = 20):
        points = self.points
        # mask = self.mask
        self.mask = mask
        if mask.any():
            
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            simplified_contours = []
            for contour in contours:
                simplified_contour = cv2.approxPolyDP(contour, tolerance, True)
                if simplified_contour.shape[0] > 1:
                    simplified_contour = simplified_contour.squeeze()
                    simplified_contours.append(simplified_contour.tolist())

            points = [shape for shape in simplified_contours if len(shape) >= 3]
            points = sorted(points, key=self.calculate_area, reverse=True)
            largest_area = max(points, key=self.calculate_area)
            largest_area_value = self.calculate_area(largest_area)


            points = [shape for shape in points if (self.calculate_area(shape) / largest_area_value) * 100 >= 0.75]
            self.points = points[0]
            return points
        else:
            raise Exception('>>> Mask not initiated <<<')
    
    
    def save_masked_image(self):
        
        if self.image is None:
            raise Exception('>>>>> Image not initiated <<<<<<')
            
            
        image_rgb = self.image
        mask = self.mask
        
        mask = resize_binary_mask(mask, (image_rgb.shape[1], image_rgb.shape[0]))

        xmin, ymin, xmax, ymax = annotations[0], annotations[1], annotations[2], annotations[3]

        inv_final_mask = ~mask

        test_image = image_rgb.copy()



        test_image_arr = np.array(test_image)
        black_image = np.ones(test_image_arr.shape)*0

        masked_out_image = np.copy(test_image_arr)
        # print(inv_final_mask.shape)
        # print(test_image_arr.shape)
        masked_out_image[inv_final_mask] = [0, 0, 0]
        masked_out_image = masked_out_image[ymin:ymax, xmin:xmax]
        self.masked_image = masked_out_image

    
    
    
        
    def find_left_upper_right_down(self):
        if self.points is None:
            raise Exception('TypeError: Points list is None')
        points = self.points
        if not points:
            warnings.warn('>>> WARNING : list of points is empty. Please Check <<<')
            return None, None
        
        left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
        right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]
        
        self.left = left_upper
        self.right = right_down
        self.annotations = [self.left[0], self.left[1], self.right[0], self.right[1]]

        
        
    def crop_image(self, image, mask, get_annotations = False, save_path = None):
        self.get_image_annotations(image, mask)
        if get_annotations == True:
            return self.annotations
        
        self.save_masked_image()
        
        if save_path == None:
            return self.masked_image 
        else:
            if self.masked_image:
                print(f'>>>>>>>> Saving image at {save_path} <<<<<<<<<<<<')
                cv2.imwrite(save_path, self.masked_image)
            else:
                raise Exception('>>>>>>> Masked Image is NoneType <<<<<<<<<')
                
                
    def get_image_annotations(self, image, mask):
        if type(image) == str:
            self.image = cv2.imread(image)
        self.image = image 
        
        boolean_mask = mask>0.5
        boolean_mask = boolean_mask.astype(np.uint8)
        self.mask = boolean_mask
        self.maskProcessor()
        self.find_left_upper_right_down()
        return self.annotations
        
        