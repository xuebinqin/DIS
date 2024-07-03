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
    
    def resize_binary_mask(self, mask,size):
        return cv2.resize(mask,size, cv2.INTER_LINEAR) 
    
    def maskedImage(self,image_rgb, mask, annotations = None):
    
        if annotations is None:
            points = self.maskProcessor(mask, 20)
            annotations = self.find_left_upper_right_down(points)
            print(annotations)
        
        mask = self.resize_binary_mask(mask, (image_rgb.shape[1], image_rgb.shape[0]))

        xmin, ymin, xmax, ymax = annotations[0], annotations[1], annotations[2], annotations[3]
        inv_final_mask = ~mask
        test_image = image_rgb.copy()
        test_image_arr = np.array(test_image)
        masked_out_image = np.copy(test_image_arr)
        masked_out_image[inv_final_mask] = [0, 0, 0]
        masked_out_image = masked_out_image[ymin:ymax, xmin:xmax]
        masked_image = masked_out_image
        return masked_image 
    
    
    
        
    def find_left_upper_right_down(self, points):
        if not points:
            warnings.warn('>>> WARNING : list of points is empty. Please Check <<<')
            return None, None
        
        left_upper = [min(points, key=lambda point: point[0])[0], min(points, key=lambda point: point[1])[1]]
        right_down = [max(points, key=lambda point: point[0])[0], max(points, key=lambda point: point[1])[1]]
        annotations = [left_upper[0], left_upper[1], right_down[0], right_down[1]]
        return annotations
        
        
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
                
    
    def get_xyxy(self, points_list):
        if points_list is None:
            vertices_list = self.points 
        else:
            vertices_list = points_list 
        
        bounding_boxes = []
        for entry in vertices_list:
            bounding_boxes.append(self.find_left_upper_right_down(entry))

        return bounding_boxes
        
