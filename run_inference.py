import time
import cv2 
from openvino.inference_engine import IENetwork,IEPlugin
import numpy as np
import util

image_data = input_img = util.img.imread("./780.jpg")
output_img = "drawn.jpg"
def check_result(y):
    pass

def preprocess(frame):
    pass

def postprocess(mask_vals, pixel_scores):
    def resize(img):
        return util.img.resize(img, size = (w, h), 
            interpolation = cv2.INTER_NEAREST)
            
    def get_bboxes(mask):
        return mask_to_bboxes(mask, image_data.shape)
            
    def draw_bboxes(img, bboxes, color):
        for bbox in bboxes:
            
            points = np.reshape(bbox, [len(bbox)//2, 2])
            cnts = util.img.points_to_contours(points)
            img = util.img.draw_contours(img, contours = cnts, 
                idx = -1, color = color, border_width = 1)
        util.img.imwrite(output_img,img)
    
    w = 640
    h = 480
    image_idx = 0
    pixel_score = pixel_scores[image_idx, ...]
    mask = mask_vals[image_idx, ...]
    
    bboxes_det = get_bboxes(mask)
    
    mask = resize(mask)
    pixel_score = resize(pixel_score)
    
    
            
    draw_bboxes(image_data, bboxes_det, util.img.COLOR_RGB_YELLOW)



def run():
    #global plugin used for loading model to MYRIAD Device
    target_device ="CPU" #  CPU
    plugin = IEPlugin(target_device)  

    # assuming IR Files is: "./text_detection.xml" & "./text_detection.bin"
    # Load the model.
    # root="./text_detection.%s"
    model = "./inference_graph.%s"

    num_requests = 1
    ######### MODEL 1 ###########
    net = IENetwork(model%'xml',model%'bin')
    device_model = plugin.load(network = net,num_requests=num_requests)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    
    # get the input image
    #input_img = util.img.imread("../ch8_test_images/7.jpg")
    print(input_img.shape)
    # preprocess the inputs
    preprocessed_img = input_img#preprocess(input_img)
    
    # Run on NCS
    device_model.infer({input_blob: preprocessed_img})
    
    # Get the outputs
    # print("",result[out_blob].shape)
    result = device_model.requests[0].outputs

    # postprocess the outputs
    output = result#postprocess(result)
    
    for i in output:       
        

        if(i=='test/Reshape_6/Transpose'):
            output[i] = output[i][:,1,:,:,:]
        else:
            output[i] = output[i][:,1,:,:]

    
    pixel_scores = output['test/Reshape_1/Transpose']
    link_scores = output['test/Reshape_6/Transpose']
    #pixel_scores = output['test/strided_slice_3/Squeeze_shrink']
    #link_scores = output['test/strided_slice_4/Squeeze_shrink/Transpose']
    mask_vals = tf_decode_score_map_to_mask_in_batch(
                pixel_scores, link_scores)

    postprocess(mask_vals, pixel_scores)

    # check the output
    check_result(output)
    
    





def tf_decode_score_map_to_mask_in_batch(pixel_cls_scores, pixel_link_scores):
    masks = decode_batch(pixel_cls_scores, pixel_link_scores)
    b, h, w = pixel_cls_scores.shape
    masks = masks.reshape((b, h, w))
    return masks

def decode_batch(pixel_cls_scores, pixel_link_scores, 
                 pixel_conf_threshold = None, link_conf_threshold = None):
    
    pixel_conf_threshold = 0.5
    link_conf_threshold = 0.7
    #print(pixel_cls_scores, pixel_link_scores)
    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    
    for image_idx in range(batch_size):

        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :, :]
           
        mask = decode_image_by_join(
            image_pos_pixel_scores, image_pos_link_scores, 
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)
    return np.asarray(batch_mask, np.int32)


def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y):
    return get_neighbours_8(x, y)
    
def get_neighbours_fn():
    return get_neighbours_8, 8



def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;
    

    
def decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    
    points = zip(*np.where(pixel_mask==True))

    h, w = np.shape(pixel_mask)
    
    group_mask = dict.fromkeys(points, -1)
    
    points = zip(*np.where(pixel_mask==True))
    
    def find_parent(point):
        return group_mask[point]
        
    def set_parent(point, parent):
        group_mask[point] = parent
        
    def is_root(point):
        return find_parent(point) == -1
    
    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        
        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)
            
        return root
        
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        
        if root1 != root2:
            set_parent(root1, root2)
        
    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        
        mask = np.zeros_like(pixel_mask, dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask
    
    # join by link
    for point in points:
        
        y, x = point
        neighbours = get_neighbours(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):

            if is_valid_cord(nx, ny, w, h):
#                 reversed_neighbours = get_neighbours(nx, ny)
#                 reversed_idx = reversed_neighbours.index((x, y))
                link_value = link_mask[y, x, n_idx]# and link_mask[ny, nx, reversed_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:

                    join(point, (ny, nx))

    points = zip(*np.where(pixel_mask==True))
    mask = get_all()
    
    return mask






def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

# @util.dec.print_calling_in_short
# @util.dec.timeit
def mask_to_bboxes(mask, image_shape =  None, min_area = None, 
                   min_height = None, min_aspect_ratio = None):
    
    feed_shape = [480, 640]
    
    if image_shape is None:
        image_shape = feed_shape
        
    image_h, image_w = image_shape[0:2]
    
    if min_area is None:
        min_area = 20
        
    if min_height is None:
        min_height = 1
    bboxes = []
    
    max_bbox_idx = int(mask.max())
    mask = util.img.resize(img = mask, size = (image_w, image_h), 
                           interpolation = cv2.INTER_NEAREST)
    

    for bbox_idx in range(1, max_bbox_idx):
        bbox_mask = mask == bbox_idx

        cnts = util.img.find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        segm_mask = []
        for pt in cnt:
            segm_mask.append(pt[0][0])
            segm_mask.append(pt[0][1])
        segm_mask = np.array(segm_mask)
        
        rect, rect_area = min_area_rect(cnt)
        
        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue
        
        if rect_area < min_area:
            continue
        
#         if max(w, h) * 1.0 / min(w, h) < 2:
#             continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        
        #bboxes.append(segm_mask)
        
    return bboxes






    
if __name__ == "__main__":
    run()
    print("Done")