import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import scipy as sc
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics import FastSAM

from ultralyticsplus import YOLO
import matplotlib.patches as patches
from scipy.signal import find_peaks, savgol_filter

from scipy.spatial import ConvexHull



import time

# HoughLines threshold
HOUGHLINE_RHO=0.5
HOUGHLINE_THR=100
BBOX_MARGIN=0
# 특정 경로
BASE_PATH = '/datalakes/0000/rapid_kit_custom/test/images'

YOLO_MODEL_COUNT = YOLO('/home/jeonggyu/rapid_kit_detection/models/detection/best_33.pt')
YOLO_MODEL = YOLO('/home/jeonggyu/rapid_kit_detection/models/detection/best_33.pt')
YOLO_MODEL_OBB = YOLO('/home/jeonggyu/rapid_kit_detection/models/detection/best_obb_4.pt')

def remove_duplicate_coordinates(coordinates, threshold=1, distance_type='euclidean'):
    flag = 0
    cleaned_coordinates = []
    cleaned_coordinates.append(coordinates[0])
    for i in range(1,len(coordinates) - 1):
        if flag:
            flag=0
            continue
        x1, y1 = coordinates[i - 1]
        x2, y2 = coordinates[i]
        x3, y3 = coordinates[i + 1]
        if distance_type == 'euclidean':
            distance1 = np.linalg.norm(coordinates[i]-coordinates[i - 1])
            distance2 = np.linalg.norm(coordinates[i+1]-coordinates[i])
        elif distance_type == 'manhattan':
            distance1 = abs(x2 - x1) + abs(y2 - y1)
            distance2 = abs(x3 - x2) + abs(y3 - y2)

        if distance1 < threshold and distance2 < threshold:
            cleaned_coordinates.append(coordinates[i+1])
            flag = 1
        else:
            cleaned_coordinates.append(coordinates[i])
            flag = 0
    #cleaned_coordinates.append(coordinates[-1])  # Add the last coordinate
    return np.array(cleaned_coordinates)

def find_steep_change_indices(coords,topk=1,window_size = 5):
    window_size = (window_size // 2) * 2 + 1

    # 좌표를 NumPy 배열로 변환
    coords = np.array(coords)

    # 각 선분의 기울기를 계산
    #slopes = np.diff(coords[:, 1]) / np.diff(coords[:, 0])
    radian = np.arctan2(np.diff(coords[:, 1]),np.diff(coords[:, 0]))
    #radian = np.arctan2(np.abs(np.diff(coords[:, 1])),np.abs(np.diff(coords[:, 0])))
    #print(f'slopes : {slopes}\n')
    #print(f'radian : {radian}\n')

    radian_changes = np.abs(np.diff(radian))
    #print(radian_changes)
    #print(np.mean(radian_changes),np.std(radian_changes))

    steep_change_indices = []
    for i in range(window_size//2, len(radian_changes) - window_size//2):  # 시작과 끝 지점을 벗어나지 않도록 인덱스 범위 설정
        sum_changes = sum(radian_changes[i - window_size//2:i + window_size//2+1])  # 인접한 5개의 라디안 변화량의 합 계산
        if sum_changes > (np.mean(radian_changes)*window_size + np.std(radian_changes)):  # 합이 평균과 표준편차를 곱한 값보다 큰지 확인
            steep_change_indices.append(i+1)

    # 변화량이 급격한 순서대로 정렬
    steep_change_indices.sort(key=lambda x: radian_changes[x-1], reverse=True)
    #print(steep_change_indices)
    # 가장 급격한 순서대로 topk개의 인덱스 반환
    return steep_change_indices[:topk]

# 패턴의 길이를 찾는 함수
def find_pattern_length(arr):
    arr_concat = np.array(arr[:,0]*1000+arr[:,1])
    return len(arr) - len(np.unique(arr_concat))

# 인접한 점들 사이에 새로운 점을 추가하는 함수
def interpolate_points(points, num_interpolated_points):
    interpolated_points = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        for j in range(1, num_interpolated_points + 1):
            # 새로운 점 추가 (start_point과 end_point 사이에)
            x = start_point[0] + (end_point[0] - start_point[0]) * j / (num_interpolated_points + 1)
            y = start_point[1] + (end_point[1] - start_point[1]) * j / (num_interpolated_points + 1)
            interpolated_points.append([x, y])
    return np.array(interpolated_points)

def remove_noise(coordinates, threshold_size):
    # Convex Hull을 이용하여 주어진 좌표들을 둘러싸는 다각형을 찾음
    hull = ConvexHull(coordinates)
    boundary_points = coordinates[hull.vertices]

    # 주어진 좌표들 중 경계 점과의 거리를 계산
    distances = np.linalg.norm(coordinates[:, None] - boundary_points, axis=-1)
    min_distances = np.min(distances, axis=1)

    # 크기 기반 필터링을 사용하여 노이즈 제거
    filtered_coordinates = coordinates[min_distances <= threshold_size]

    return filtered_coordinates


def crop_image_from_yolo_result(result, target_class=1, bgr=True):
    if bgr:
        image_bgr = result.orig_img
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = result.orig_img
    
    h, w = image_rgb.shape[0], image_rgb.shape[1]
    
    boxes = result.boxes

    cropped_imgs = list()
    for box in boxes:
        if box.cls==target_class:
            x1,y1,x2,y2 = box.xyxy.int().tolist()[0]
            x1,y1,x2,y2 = np.max([0,x1-BBOX_MARGIN]),np.max([0,y1-BBOX_MARGIN]),np.min([w,x2+BBOX_MARGIN]),np.min([h,y2+BBOX_MARGIN])
            #print(x1,y1,x2,y2)
            cropped_imgs.append({'img': image_rgb[y1:y2,x1:x2], 'center': [int((x1+x2)/2),int((y1+y2)/2)]})

    cropped_imgs = sorted(cropped_imgs, key=lambda xy: xy['center'][0]+xy['center'][1])  # x 좌표를 기준으로 정렬
    return image_rgb, cropped_imgs

def c_value_line_from_img(img, strip_crop_ratio=1):

    # 이미지의 높이와 너비를 가져옵니다.
    height, width = img.shape[:2]

    if height > width :
        # crop할 영역의 높이와 너비를 계산합니다.
        crop_height = int(height*0.9)
        crop_width = int(width * strip_crop_ratio)  
    else:
        # crop할 영역의 높이와 너비를 계산합니다.
        crop_height = int(height * strip_crop_ratio)
        crop_width = int(width*0.9)


    # 좌상단 점을 계산합니다.
    start_y = (height - crop_height) // 2
    start_x = (width - crop_width) // 2

    # 이미지를 crop합니다.
    cropped_image = img[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # 제로 패딩된 이미지를 만듭니다.
    padded_image = np.zeros_like(img)+255
    padded_image[start_y:start_y + crop_height, start_x:start_x + crop_width] = cropped_image
    # 평균을 구합니다.
    if padded_image.shape[0] > padded_image.shape[1]:
        value_line = padded_image.mean(axis=1)
    else:
        value_line = padded_image.mean(axis=0)

    colors = []
    for i in range(value_line.shape[0]):
        colors.append(f'#{int(value_line[i][0]):02x}{int(value_line[i][1]):02x}{int(value_line[i][2]):02x}')

    return value_line, colors

def find_peaks_with_sg_filter(vector, window_length=11, polyorder=2, distance=None):
    smoothed = savgol_filter(vector, window_length, polyorder)
    peaks, _ = find_peaks(smoothed,distance=distance)
    return peaks, smoothed

# 인접한 점들 사이에 새로운 점을 추가하는 함수
def interpolate_points(points, num_interpolated_points):
    interpolated_points = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        for j in range(1, num_interpolated_points + 1):
            # 새로운 점 추가 (start_point과 end_point 사이에)
            x = start_point[0] + (end_point[0] - start_point[0]) * j / (num_interpolated_points + 1)
            y = start_point[1] + (end_point[1] - start_point[1]) * j / (num_interpolated_points + 1)
            interpolated_points.append([x, y])
    return np.array(interpolated_points)

def perspective_transform(input_img,predictor):
    input_heigth = input_img.shape[0]
    input_width= input_img.shape[1]
    
    # Set image
    predictor.set_image(input_img)  # set with image file
    #sam_results = predictor(points=[input_img.shape[0]//2,input_img[1]//2],labels=[0])
    #sam_results = predictor(bboxes=[])
    sam_results = predictor()

    # Reset image
    predictor.reset_image()
    

    largest_segment_size = 0
    largest_segment_idx = 0
    for i in range(len(sam_results[0])):
        xy_ = sam_results[0].masks[i].xy[0]
        seg_width = xy_[:,0].max()-xy_[:,0].min()
        seg_heigth = xy_[:,1].max()-xy_[:,1].min()
        seg_size = seg_heigth*seg_width
        #print(seg_width,seg_heigth,seg_size)
        if seg_size > largest_segment_size: 
            largest_segment_size = seg_size 
            largest_segment_idx = i

    xy = sam_results[0].masks[largest_segment_idx].xy[0]
    matrix = np.zeros((2,2))
    org_x = int((xy[:,0].max()+xy[:,0].min())/2)
    org_y = int((xy[:,1].max()+xy[:,1].min())/2)
    #print(org_x,org_y)

    xy[:,0] = xy[:,0] - org_x
    xy[:,1] = xy[:,1] - org_y
    
    matrix = np.zeros((2,2))
    for i in range(len(xy)):
        if i == 0 :
            pass
        else: 
            dist = math.sqrt((xy[i][0] - xy[i-1][0])**2 + (xy[i][1] - xy[i-1][1])**2)
            xx_=xy[i][0]*xy[i][0]
            yy_=xy[i][1]*xy[i][1]
            xy_=xy[i][0]*xy[i][1]
            matrix[0][0]+= xx_*dist
            matrix[1][1]+= yy_*dist
            matrix[1][0]-= xy_*dist
            matrix[0][1]-= xy_*dist
    
    w, vl= sc.linalg.eig(matrix)

    # 새로운 축을 기준으로 각도 계산
    angle = np.arctan2(vl[0,0], vl[0,1])

    # 회전 행렬 생성
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 각 점을 회전시킴
    rotated_points = np.dot(xy, rotation_matrix.T)  # 전치(transpose)는 회전 행렬을 사용하기 위해 필요합니다.
 
 
    num_interpolated_points = 3
    rotated_points_interp = interpolate_points(rotated_points, num_interpolated_points)
 

    # x좌표를 기준으로 정렬하여 가장 좌측과 우측에 있는 점과 인덱스 찾기
    leftmost_index, leftmost_point = min(enumerate(rotated_points_interp), key=lambda x: x[1][0])
    rightmost_index, rightmost_point = max(enumerate(rotated_points_interp), key=lambda x: x[1][0])

    # y좌표를 기준으로 정렬하여 가장 위와 아래에 있는 점과 인덱스 찾기
    topmost_index, topmost_point = max(enumerate(rotated_points_interp), key=lambda x: x[1][1])
    bottommost_index, bottommost_point = min(enumerate(rotated_points_interp), key=lambda x: x[1][1])

    most_4pt = np.array([leftmost_point,rightmost_point,topmost_point,bottommost_point])
    most_4pt_index = np.array([leftmost_index,rightmost_index,topmost_index,bottommost_index])

    edge_4pt = most_4pt
    edge_4pt_dist = [0,0,0,0]

    for ind, pt in enumerate(most_4pt):
        cur_dist = math.sqrt((pt[0])**2 + (pt[1])**2)
        edge_4pt_dist[ind] = cur_dist
        
    for ind, xy in enumerate(rotated_points_interp):
        cur_dist = math.sqrt((xy[0])**2 + (xy[1])**2)
        if ind < np.min([most_4pt_index[0]+3,len(rotated_points_interp)-1]) and ind > np.max([most_4pt_index[0]-3,0]):
            if edge_4pt_dist[0] < cur_dist:
                edge_4pt_dist[0] = cur_dist
                edge_4pt[0] = [xy[0],xy[1]]
        elif ind < np.min([most_4pt_index[1]+3,len(rotated_points_interp)-1]) and ind > np.max([most_4pt_index[1]-3,0]):
            if edge_4pt_dist[1] < cur_dist:
                edge_4pt_dist[1] = cur_dist
                edge_4pt[1] = [xy[0],xy[1]]
        elif ind < np.min([most_4pt_index[2]+3,len(rotated_points_interp)-1]) and ind > np.max([most_4pt_index[2]-3,0]):
            if edge_4pt_dist[2] < cur_dist:
                edge_4pt_dist[2] = cur_dist
                edge_4pt[2] = [xy[0],xy[1]]
        elif ind < np.min([most_4pt_index[3]+3,len(rotated_points_interp)-1]) and ind > np.max([most_4pt_index[3]-3,0]):
            if edge_4pt_dist[3] < cur_dist:
                edge_4pt_dist[3] = cur_dist
                edge_4pt[3] = [xy[0],xy[1]]
    edge_4pt = np.array(edge_4pt)

    restored_points = np.dot(edge_4pt, rotation_matrix)
    restored_points[:,0] += org_x
    restored_points[:,1] += org_y

    pts = restored_points
    sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

    topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
    bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    #print(mtrx)
    # 원근 변환 적용
    result = cv2.warpPerspective(input_img, mtrx, (int(width), int(height)))
    
    return result

def perspective_transform_v2(input_img,predictor):
    # Set image
    predictor.set_image(input_img)  # set with image file
    #sam_results = predictor(points=[input_img.shape[0]//2,input_img[1]//2],labels=[0])
    #sam_results = predictor(bboxes=[])
    sam_results = predictor()

    # Reset image
    predictor.reset_image()

    largest_segment_size = 0
    largest_segment_idx = 0
    for i in range(len(sam_results[0])):
        xy_ = sam_results[0].masks[i].xy[0]
        seg_width = xy_[:,0].max()-xy_[:,0].min()
        seg_heigth = xy_[:,1].max()-xy_[:,1].min()
        seg_size = seg_heigth*seg_width
        #print(seg_width,seg_heigth,seg_size)
        if seg_size > largest_segment_size: 
            largest_segment_size = seg_size 
            largest_segment_idx = i

    xy = sam_results[0].masks[largest_segment_idx].xy[0]
    matrix = np.zeros((2,2))
    org_x = int((xy[:,0].max()+xy[:,0].min())/2)
    org_y = int((xy[:,1].max()+xy[:,1].min())/2)
    #print(org_x,org_y)

    xy[:,0] = xy[:,0] - org_x
    xy[:,1] = xy[:,1] - org_y
    
    matrix = np.zeros((2,2))
    for i in range(len(xy)):
        if i == 0 :
            pass
        else: 
            dist = math.sqrt((xy[i][0] - xy[i-1][0])**2 + (xy[i][1] - xy[i-1][1])**2)
            xx_=xy[i][0]*xy[i][0]
            yy_=xy[i][1]*xy[i][1]
            xy_=xy[i][0]*xy[i][1]
            matrix[0][0]+= xx_*dist
            matrix[1][1]+= yy_*dist
            matrix[1][0]-= xy_*dist
            matrix[0][1]-= xy_*dist
    
    w, vl= sc.linalg.eig(matrix)

    # 새로운 축을 기준으로 각도 계산
    angle = np.arctan2(vl[0,0], vl[0,1])

    # 회전 행렬 생성
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 각 점을 회전시킴
    rotated_points = np.dot(xy, rotation_matrix.T)  # 전치(transpose)는 회전 행렬을 사용하기 위해 필요합니다.
 
 
 
    num_interpolated_points = 100
    sparse_divide_num=10

    cleaned_coordinates = remove_duplicate_coordinates(rotated_points, threshold=2)
    print(f'{len(rotated_points)-len(cleaned_coordinates)} points removed')

    rotated_points_flip = np.flip(cleaned_coordinates,axis=0)
    filtered_coordinates = rotated_points_flip[(rotated_points_flip[:, 0] > 0) & (rotated_points_flip[:, 1] > 0)]
    closest_to_zero = filtered_coordinates[np.argmin(filtered_coordinates[:, 1])]
    closest_to_zero_ind = np.where(rotated_points_flip==closest_to_zero)[0][0]
    reodered_indices = np.arange(0,len(rotated_points_flip))
    reodered_indices = (reodered_indices+closest_to_zero_ind)%len(rotated_points_flip)
    reodered_arr = rotated_points_flip[reodered_indices]
    #print(closest_to_zero_ind)

    x_coords = reodered_arr[:, 0]
    y_coords = reodered_arr[:, 1]

    # Apply Savitzky-Golay filter to x and y coordinates
    filtered_x = savgol_filter(x_coords, window_length=10, polyorder=1)
    filtered_y = savgol_filter(y_coords, window_length=10, polyorder=1)

    reodered_cleaned_coordinates = np.column_stack((filtered_x, filtered_y))

    #print(f'{len(reodered_cleaned_coordinates)} points and ')
    cleaned_coordinates = remove_duplicate_coordinates(reodered_cleaned_coordinates, threshold=5)
    print(f'{len(reodered_cleaned_coordinates)-len(cleaned_coordinates)} points removed')


    reodered_arr_interp = interpolate_points(cleaned_coordinates, num_interpolated_points)
    print(f'{len(cleaned_coordinates)-len(reodered_arr_interp)} points added')

    #print(f'{len(reodered_arr_interp)} points and ')
    cleaned_coordinates = remove_duplicate_coordinates(reodered_arr_interp, threshold=5)
    print(f'{len(reodered_arr_interp)-len(cleaned_coordinates)} points removed')
    # 인덱스를 n씩 건너뛰면서 값을 출력
    sparse_coordinates = cleaned_coordinates[::sparse_divide_num]
    print(f'{len(cleaned_coordinates)-len(sparse_coordinates)} points removed')
    
    print(f'{len(sparse_coordinates)} points remain')

    cleaned_coordinates_circ = sparse_coordinates

    # Divide cleaned_coordinates into quadrants
    q1 = cleaned_coordinates_circ[np.logical_and(cleaned_coordinates_circ[:, 0] > 0, cleaned_coordinates_circ[:, 1] > 0)]
    q2 = cleaned_coordinates_circ[np.logical_and(cleaned_coordinates_circ[:, 0] < 0, cleaned_coordinates_circ[:, 1] > 0)]
    q3 = cleaned_coordinates_circ[np.logical_and(cleaned_coordinates_circ[:, 0] < 0, cleaned_coordinates_circ[:, 1] < 0)]
    q4 = cleaned_coordinates_circ[np.logical_and(cleaned_coordinates_circ[:, 0] > 0, cleaned_coordinates_circ[:, 1] < 0)]

    q1_unique = q1
    q2_unique = q2
    q3_unique = q3
    q4_unique = q4

    ratio= 0.2
    # Find steep change indices for each quadrant
    indices_q1 = find_steep_change_indices(q1_unique, window_size=int(len(q1_unique)*ratio))
    indices_q2 = find_steep_change_indices(q2_unique, window_size=int(len(q2_unique)*ratio))
    indices_q3 = find_steep_change_indices(q3_unique, window_size=int(len(q3_unique)*ratio))
    indices_q4 = find_steep_change_indices(q4_unique, window_size=int(len(q4_unique)*ratio))
    #indices_q1 = find_steep_change_indices(q1_unique, window_size=sparse_divide_num*5)
    #indices_q2 = find_steep_change_indices(q2_unique, window_size=sparse_divide_num*5)
    #indices_q3 = find_steep_change_indices(q3_unique, window_size=sparse_divide_num*5)
    #indices_q4 = find_steep_change_indices(q4_unique, window_size=sparse_divide_num*5)
    #print(len(q1_unique), indices_q1, int(len(q1_unique)*ratio))
    #print(len(q2_unique), indices_q2, int(len(q2_unique)*ratio))
    #print(len(q3_unique), indices_q3, int(len(q3_unique)*ratio))
    #print(len(q4_unique), indices_q4, int(len(q4_unique)*ratio))

    # Update edge_4pt with the calculated indices
    edge_4pt = np.array([q1_unique[indices_q1[0]], q2_unique[indices_q2[0]], q3_unique[indices_q3[0]], q4_unique[indices_q4[0]]])

    print(edge_4pt)

    restored_points = np.dot(edge_4pt, rotation_matrix)
    restored_points[:,0] += org_x
    restored_points[:,1] += org_y

    pts = restored_points
    sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

    topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
    bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    #print(mtrx)
    # 원근 변환 적용
    result = cv2.warpPerspective(input_img, mtrx, (int(width), int(height)))
    
    return result

def perspective_transform_v3(input_obj,predictor,orig_img):
    input_img = input_obj['img']
    num_interpolated_points = 100

    # Set image
    predictor.set_image(input_img)  # set with image file
    sam_results = predictor()

    # Reset image
    predictor.reset_image()
    
    largest_segment_size = 0
    largest_segment_idx = 0
    for i in range(len(sam_results[0])):
        xy = sam_results[0].masks[i].xy[0]

        seg_width = xy[:,0].max()-xy[:,0].min()
        seg_heigth = xy[:,1].max()-xy[:,1].min()
        seg_size = seg_heigth*seg_width
        
        shp = sam_results[0].masks[i].data.shape
        
        if sam_results[0].masks[i].data[shp[0]-1][shp[1]//2-1][shp[2]//2-1]:
            print(seg_width,seg_heigth,seg_size)
            if seg_size > largest_segment_size: 
                largest_segment_size = seg_size 
                largest_segment_idx = i
    print(largest_segment_idx)

    xy = sam_results[0].masks[largest_segment_idx].xy[0]

    #xy = interpolate_points(xy, num_interpolated_points)
    '''
    matrix = np.zeros((2,2))
    org_x = int((xy[:,0].max()+xy[:,0].min())/2)
    org_y = int((xy[:,1].max()+xy[:,1].min())/2)

    xy[:,0] = xy[:,0] - org_x
    xy[:,1] = xy[:,1] - org_y
    
    matrix = np.zeros((2,2))
    for i in range(len(xy)):
        if i == 0 :
            pass
        else: 
            dist = math.sqrt((xy[i][0] - xy[i-1][0])**2 + (xy[i][1] - xy[i-1][1])**2)
            xx_=xy[i][0]*xy[i][0]
            yy_=xy[i][1]*xy[i][1]
            xy_=xy[i][0]*xy[i][1]
            matrix[0][0]+= xx_*dist
            matrix[1][1]+= yy_*dist
            matrix[1][0]-= xy_*dist
            matrix[0][1]-= xy_*dist
    
    w, vl= sc.linalg.eig(matrix)

    # 새로운 축을 기준으로 각도 계산
    angle = np.arctan2(vl[0,0], vl[0,1])

    # 회전 행렬 생성
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # 각 점을 회전시킴
    rotated_points = np.dot(xy, rotation_matrix.T)  # 전치(transpose)는 회전 행렬을 사용하기 위해 필요합니다.
    
    '''
    rotated_points=xy
    search_size = len(rotated_points)//16

    caclulated_arr=[]
    rotated_points_circ = np.concatenate((rotated_points[-search_size:],rotated_points,rotated_points[:search_size]))
    for i in range(search_size,len(rotated_points_circ)-search_size):
        ldist = math.sqrt((rotated_points_circ[i][0] - rotated_points_circ[i-search_size][0])**2 + (rotated_points_circ[i][1] - rotated_points_circ[i-search_size][1])**2)
        rdist = math.sqrt((rotated_points_circ[i+search_size][0] - rotated_points_circ[i][0])**2 + (rotated_points_circ[i+search_size][1] - rotated_points_circ[i][1])**2)
        lx=(rotated_points_circ[i][0] - rotated_points_circ[i-search_size][0])/ldist
        ly=(rotated_points_circ[i][1] - rotated_points_circ[i-search_size][1])/ldist
        rx=(rotated_points_circ[i][0] - rotated_points_circ[i+search_size][0])/rdist
        ry=(rotated_points_circ[i][1] - rotated_points_circ[i+search_size][1])/rdist
        elem=[lx+rx,ly+ry]
        caclulated_arr.append(elem)
    caclulated_arr = np.array(caclulated_arr)

    #curv_v_mean = caclulated_arr.mean(axis=0)
    #curv_v_std = caclulated_arr.std(axis=0)
    #condition_xy=curv_v_mean+curv_v_std*3

    target_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])

    filtered_coordinates = []

    # 모든 점들 간의 거리 계산
    distances = np.linalg.norm(caclulated_arr[:, None, :] - target_points[None, :, :], axis=-1)

    # 각 점마다 가장 가까운 점의 인덱스 찾기
    closest_indices = np.argmin(distances, axis=0)

    # 가장 가까운 점들 찾기
    for i, idx in enumerate(closest_indices):
        filtered_coordinates.append(caclulated_arr[idx])

    filtered_coordinates = np.array(filtered_coordinates)

    mask = np.isin(caclulated_arr,filtered_coordinates)
    indices = np.where(mask)[0]

    edge_4pt = rotated_points[indices]

    print(edge_4pt)
    '''
    restored_points = np.dot(edge_4pt, rotation_matrix)
    restored_points[:,0] += org_x
    restored_points[:,1] += org_y
    pts = restored_points
    '''
    pts = edge_4pt

    sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

    topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
    bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    #print(mtrx)
    
    #input img에 expand 적용하면?
    cy,cx = input_obj['center']
    cw,ch,_ = input_obj['img'].shape
    dx = int(cw/2*0.1)
    dy = int(ch/2*0.1)

    print(cx,cy,dx,dy)
    print(orig_img.shape)
    orig_w, orig_h,_ = orig_img.shape
    ratio = 12
    expanded_cropped_img = orig_img[np.max([0,cx-dx*ratio]):np.min([orig_w,cx+dx*ratio]),np.max([0,cy-dy*ratio]):np.min([orig_h,cy+dy*ratio]),:]
    expanded_cropped_img = cv2.cvtColor(expanded_cropped_img, cv2.COLOR_BGR2RGB)

    # 원근 변환 적용
    #result = cv2.warpPerspective(input_img, mtrx, (int(width), int(height)))
    result = cv2.warpPerspective(expanded_cropped_img, mtrx, (int(width*(ratio/10)), int(height*(ratio/10))))
    
    return result


def perspective_transform_v4(input_obj, FSAM, orig_img):
    start_time = time.time()

    input_img = input_obj['img']
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)

    num_interpolated_points = 100

    sam_results = FSAM(input_img, device='cuda:1', retina_masks=True, imgsz=1024, conf=0.25, save=True, project='fsam_outputs')

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 1 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    largest_segment_size = 0
    largest_segment_idx = 0
    for i in range(len(sam_results[0])):
        xy = sam_results[0].masks[i].xy[0]

        seg_width = xy[:,0].max()-xy[:,0].min()
        seg_heigth = xy[:,1].max()-xy[:,1].min()
        seg_size = seg_heigth*seg_width
        
        shp = sam_results[0].masks[i].data.shape
        
        if sam_results[0].masks[i].data[shp[0]-1][shp[1]//2-1][shp[2]//2-1]:
            print(seg_width,seg_heigth,seg_size)
            if seg_size > largest_segment_size: 
                largest_segment_size = seg_size 
                largest_segment_idx = i
    print(largest_segment_idx)

    xy = sam_results[0].masks[largest_segment_idx].xy[0]

    xy = interpolate_points(xy, num_interpolated_points)

    #xy = remove_noise(xy, 100)

    org_x = int((xy[:, 0].max() + xy[:, 0].min()) / 2)
    org_y = int((xy[:, 1].max() + xy[:, 1].min()) / 2)

    xy[:, 0] -= org_x
    xy[:, 1] -= org_y

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 2 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    matrix = np.zeros((2, 2))
    for i in range(1, len(xy)):
        dist = np.linalg.norm(xy[i] - xy[i - 1])
        xx_ = xy[i][0] * xy[i][0]
        yy_ = xy[i][1] * xy[i][1]
        xy_ = xy[i][0] * xy[i][1]
        matrix[0][0] += xx_ * dist
        matrix[1][1] += yy_ * dist
        matrix[1][0] -= xy_ * dist
        matrix[0][1] -= xy_ * dist

    w, vl = np.linalg.eig(matrix)

    # Calculate angle
    angle = np.arctan2(vl[0, 0], vl[0, 1])

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Rotate points
    rotated_points = np.dot(xy, rotation_matrix.T)

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 3 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    search_size = len(rotated_points) // 16

    calculated_arr = []
    rotated_points_circ = np.concatenate((rotated_points[-search_size:], rotated_points, rotated_points[:search_size]))
    for i in range(search_size, len(rotated_points_circ) - search_size):
        ldist = np.linalg.norm(rotated_points_circ[i] - rotated_points_circ[i - search_size])
        rdist = np.linalg.norm(rotated_points_circ[i + search_size] - rotated_points_circ[i])
        lx = (rotated_points_circ[i][0] - rotated_points_circ[i - search_size][0]) / ldist
        ly = (rotated_points_circ[i][1] - rotated_points_circ[i - search_size][1]) / ldist
        rx = (rotated_points_circ[i][0] - rotated_points_circ[i + search_size][0]) / rdist
        ry = (rotated_points_circ[i][1] - rotated_points_circ[i + search_size][1]) / rdist
        elem = [lx + rx, ly + ry]
        calculated_arr.append(elem)
    calculated_arr = np.array(calculated_arr)

    target_points = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])

    distances = np.linalg.norm(calculated_arr[:, None, :] - target_points[None, :, :], axis=-1)
    closest_indices = np.argmin(distances, axis=0)

    filtered_coordinates = calculated_arr[closest_indices]

    mask = np.isin(calculated_arr, filtered_coordinates)
    indices = np.where(mask)[0]

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 4 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    edge_4pt = rotated_points[indices]
    print(f'edge_4pt : {edge_4pt}')
    restored_points = np.dot(edge_4pt, rotation_matrix)
    restored_points[:, 0] += org_x
    restored_points[:, 1] += org_y

    print(f'restored_points : {restored_points}')

    pts = restored_points
    sm = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])
    height = max([h1, h2])
    
    x = pts1[0,0]
    y = pts1[0,1]

    # specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    pts2 = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])
    #pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    
    '''
    cx, cy = input_obj['center']
    ch, cw, _ = input_obj['img'].shape
    print('----- ',cx,cy,cw,ch)
    dx = int(cw / 2 * 0.1)
    dy = int(ch / 2 * 0.1)
    
    orig_h, orig_w, _ = orig_img.shape
    print('----- ',orig_w,orig_h)
    ratio = 12
    print(np.max([0,cx - dx * ratio]),np.min([cx + dx * ratio,orig_w]),np.max([0,cy - dy * ratio]),np.min([cy + dy * ratio,orig_h]))
    #expanded_cropped_img = orig_img[np.clip(cx - dx * ratio, 0, orig_w):np.clip(cx + dx * ratio, 0, orig_w),
    #                     np.clip(cy - dy * ratio, 0, orig_h):np.clip(cy + dy * ratio, 0, orig_h), :]
    expanded_cropped_img = orig_img[np.max([0,cx - dx * ratio]):np.min([cx + dx * ratio,orig_w]),np.max([0,cy - dy * ratio]):np.min([cy + dy * ratio,orig_h]),:]
    
    expanded_cropped_img = cv2.cvtColor(expanded_cropped_img, cv2.COLOR_BGR2RGB)

    #result = cv2.warpPerspective(expanded_cropped_img, mtrx, (int(width * (ratio / 10)), int(height * (ratio / 10))))
    result = cv2.warpPerspective(expanded_cropped_img, mtrx, (expanded_cropped_img.shape[0],expanded_cropped_img.shape[1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    '''
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    result = cv2.warpPerspective(input_img, mtrx, (input_img.shape[1],input_img.shape[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 5 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    return result 

def perspective_transform_v5(input_obj, orig_img):
    start_time = time.time()

    input_img = input_obj['img']
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)

    results = YOLO_MODEL_OBB.predict(source=input_img, save=True, save_txt=False, conf=0.25, iou=0.7)

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 1 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    obb_list = []
    for i in range(len(results[0].obb)):
        if results[0].obb[i].cls==0:
            obb_list.append(results[0].obb[i].xyxyxyxy[0].cpu().numpy())

    pts = obb_list[0]
    sm = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])
    height = max([h1, h2])
    
    x = pts1[0,0]
    y = pts1[0,1]

    # specify output coordinates for corners of red quadrilateral in order TL, TR, BR, BL as x,
    pts2 = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])
    #pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    
    '''
    cx, cy = input_obj['center']
    ch, cw, _ = input_obj['img'].shape
    print('----- ',cx,cy,cw,ch)
    dx = int(cw / 2 * 0.1)
    dy = int(ch / 2 * 0.1)
    
    orig_h, orig_w, _ = orig_img.shape
    print('----- ',orig_w,orig_h)
    ratio = 12
    print(np.max([0,cx - dx * ratio]),np.min([cx + dx * ratio,orig_w]),np.max([0,cy - dy * ratio]),np.min([cy + dy * ratio,orig_h]))
    #expanded_cropped_img = orig_img[np.clip(cx - dx * ratio, 0, orig_w):np.clip(cx + dx * ratio, 0, orig_w),
    #                     np.clip(cy - dy * ratio, 0, orig_h):np.clip(cy + dy * ratio, 0, orig_h), :]
    expanded_cropped_img = orig_img[np.max([0,cx - dx * ratio]):np.min([cx + dx * ratio,orig_w]),np.max([0,cy - dy * ratio]):np.min([cy + dy * ratio,orig_h]),:]
    
    expanded_cropped_img = cv2.cvtColor(expanded_cropped_img, cv2.COLOR_BGR2RGB)

    #result = cv2.warpPerspective(expanded_cropped_img, mtrx, (int(width * (ratio / 10)), int(height * (ratio / 10))))
    result = cv2.warpPerspective(expanded_cropped_img, mtrx, (expanded_cropped_img.shape[0],expanded_cropped_img.shape[1]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    '''
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    result = cv2.warpPerspective(input_img, mtrx, (input_img.shape[1],input_img.shape[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    execution_time = time.time() - start_time
    print(f"- - - - - - - - - pt 5 : {execution_time}초 - - - - - - ")
    start_time = time.time()

    return result
def result_image_gen_v3(image_rgb, cropped_imgs, image_path, figsize=10):
    value_lines = list()
    colors_list=list()
    fontsize = figsize//2+5   
    strip_crop_ratio = 0.5
    
    for i in range(len(cropped_imgs)):
        value_line, colors = c_value_line_from_img(cropped_imgs[i]['img'],strip_crop_ratio=strip_crop_ratio)
        value_lines.append(value_line)
        colors_list.append(colors)
    
    plt.figure(figsize=(figsize,figsize))
    plt.imshow(image_rgb)
    for kit_ind in range(len(cropped_imgs)):
        kit_width = cropped_imgs[kit_ind]['kit_width']
        kit_heigth = cropped_imgs[kit_ind]['kit_heigth']
        kit_origin = cropped_imgs[kit_ind]['kit_origin']
        rect = patches.Rectangle((kit_origin[0], kit_origin[1]),
                                 kit_heigth, kit_width, linewidth=0.5, edgecolor='g', facecolor='None')
        plt.annotate(f"{kit_ind}", (kit_origin[0], kit_origin[1]), c='green', fontsize=fontsize*2)
        plt.gca().add_patch(rect)
    plt.show()
    file_name = image_path.split('/')[-1]
    ext = file_name.split('.')[-1]
    file_name = "".join(file_name.split('.')[:-1])
    plt.savefig(f'./outputs/{file_name}_original.{ext}')
    print(f'./outputs/{file_name}_original.{ext} (saved)')

    subplot_y = np.min([4,len(cropped_imgs)*2])
    subplot_x_kit = round(len(cropped_imgs)//(subplot_y//2)+0.51)
    subplot_x = subplot_x_kit + round(len(cropped_imgs)//(subplot_y//2)+0.51)
    print(f'subplot_x_kit : {subplot_x_kit}, subplot_x : {subplot_x}, subplot_y : {subplot_y}')
    #plt.figure(figsize=((figsize//3)*subplot_y,(figsize//3)*subplot_x))
    fontsize = np.max([fontsize-subplot_x_kit*2,8])
    for kit_ind in range(len(cropped_imgs)):
        plt.subplot(subplot_x, subplot_y, kit_ind*2+1)
        plt.imshow(cropped_imgs[kit_ind]['transformed_kit_img'])
        plt.gca().set_title(f'{kit_ind}')        

    for img_ind in range(len(cropped_imgs)):
        strip_width = np.array(cropped_imgs[img_ind]['img'].shape[:2]).min()#*0.9
        strip_heigth = np.array(cropped_imgs[img_ind]['img'].shape[:2]).max()#*0.9
        strip_center = [strip_width//2,strip_heigth//2]
        strip_center_in_kit = cropped_imgs[img_ind]['center']
        
        plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
        plt.imshow(cropped_imgs[img_ind]['img'])
        plt.gca().set_title(f'{img_ind}')        
            
        if cropped_imgs[img_ind]['img'].shape[0]>cropped_imgs[img_ind]['img'].shape[1]:
            x = np.linspace(0,1,len(value_lines[img_ind]))
            x = (x-np.min(x))/(np.max(x)-np.min(x))*(strip_width*3)
            #x_offset = strip_center[0]-(strip_width*3)/2 #offset
            #x = x+x_offset

            y_ = value_lines[img_ind].sum(axis=1)
            y = (y_-np.min(y_))/(np.max(y_)-np.min(y_))*(strip_heigth*0.6)
            #y_offset = strip_center[1]-strip_heigth*1.5
            #y = y+y_offset
            
            #bounding box
            plt.subplot(subplot_x, subplot_y, 2*img_ind+1)
            rect = patches.Rectangle((strip_center_in_kit[0]-strip_width/2, strip_center_in_kit[1]-strip_heigth/2), strip_width, strip_heigth, linewidth=0.5, edgecolor='g', facecolor='None')
            plt.gca().add_patch(rect)
            
            rotation = 0
            
        else:
            x = np.linspace(0,1,len(value_lines[img_ind]))
            x = (x-np.min(x))/(np.max(x)-np.min(x))*(strip_width*3)
            #x_offset =strip_center[0]-strip_heigth*1.5 #offset
            #x = x+x_offset

            y_ = value_lines[img_ind].sum(axis=1)
            y = (y_-np.min(y_))/(np.max(y_)-np.min(y_))*(strip_heigth*0.6)
            #y_offset = strip_center[1]-(strip_width*3)/2
            #y = y+y_offset
            
            #bounding box
            plt.subplot(subplot_x, subplot_y, 2*img_ind+1)
            rect = patches.Rectangle((strip_center_in_kit[0]-strip_heigth/2, strip_center_in_kit[1]-strip_width/2), strip_heigth, strip_width, linewidth=0.5, edgecolor='g', facecolor='None')
            plt.gca().add_patch(rect)

            rotation = 1
        
        y_t = -y_/3+255

        plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+2)
        plt.scatter(x,y_t,c=colors_list[img_ind],s=figsize//5)
        sorted_unique_arr = sorted(set(y_t))
        # 두 번째로 작은 값을 가져옵니다.
        second_smallest = sorted_unique_arr[1]
        biggest = sorted_unique_arr[-1]
        plt.ylim(np.max([second_smallest-5,0]),np.min([biggest+5,255]))
        
        peaks, smoothed = find_peaks_with_sg_filter(y_t, window_length=11, polyorder=3, distance=len(y_t)//9)
        #print(f"Peaks({len(y_t)}):{peaks}")
        #print("Smoothed peak value:", smoothed[peaks])
        #peaks = peaks[smoothed[peaks]>np.mean(smoothed)]#+np.std(smoothed)]
        peaks = peaks[(peaks>len(smoothed)*0.15) & (peaks<len(smoothed)*0.85)]
        #print(peaks)
        peaks_value = smoothed[peaks]
        #print(peaks_value)
        sorted_indices = sorted(range(len(peaks_value)), key=lambda i: peaks_value[i], reverse=True)#[:2]
        peaks = peaks[sorted_indices]
        #background_x = list()
        #background_y = list()
        for peak_ind, peak in enumerate(peaks):
            if rotation==0:
                y_peak = peak
                #print(rotation, x_peak,y_peak)
                plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
                rect = patches.Arrow(strip_width*(1-strip_crop_ratio)//2,y_peak,strip_width*strip_crop_ratio,0 ,alpha = 0.3, linewidth=0.5, edgecolor='magenta', facecolor='None')
                plt.gca().add_patch(rect)
                plt.annotate(f"{smoothed[peak]:.0f}", (strip_width//2, y_peak), c='magenta', fontsize=fontsize)
                #background_x.append(strip_center[0])
                #background_y.append(strip_center[1] - strip_heigth//2 + peak)
            else:
                x_peak = peak
                #print(rotation, x_peak,y_peak)
                plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
                rect = patches.Arrow(x_peak,strip_width*(1-strip_crop_ratio)/2,0,strip_width*strip_crop_ratio ,alpha = 0.3, linewidth=0.5, edgecolor='magenta', facecolor='None')
                plt.gca().add_patch(rect)
                plt.annotate(f"{smoothed[peak]:.0f}", (x_peak+3, strip_width//2), c='magenta', fontsize=fontsize)
                #background_x.append(strip_center[0] - strip_heigth//2 + peak)
                #background_y.append(strip_center[1])

        if rotation==0:
            plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
            plt.annotate(f"{smoothed[strip_heigth//2]:.0f}", (strip_width*0.2, strip_heigth//2), c='green', fontsize=fontsize)
            rect = patches.Arrow(strip_width*(1-strip_crop_ratio)//2,strip_heigth//2,strip_width*strip_crop_ratio,0 ,alpha = 0.3, linewidth=0.5, edgecolor='green', facecolor='None')
            plt.gca().add_patch(rect)
        else:
            plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
            plt.annotate(f"{smoothed[strip_heigth//2]:.0f}", (strip_heigth//2, strip_width*0.8), c='green', fontsize=fontsize)
            rect = patches.Arrow(strip_heigth//2,strip_width*(1-strip_crop_ratio)/2,0,strip_width*strip_crop_ratio ,alpha = 0.3, linewidth=0.5, edgecolor='green', facecolor='None')
            plt.gca().add_patch(rect)
    #return y_
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    plt.show()
    file_name = image_path.split('/')[-1]
    ext = file_name.split('.')[-1]
    file_name = "".join(file_name.split('.')[:-1])
    plt.savefig(f'./outputs/{file_name}_cropped.{ext}',bbox_inches='tight', pad_inches=1)
    print(f'./outputs/{file_name}_cropped.{ext} (saved)')

def find_closest_coordinate(coordinates, origin):
    coordinates = np.array(coordinates)
    origin = np.array(origin)
    diff = coordinates - origin
    diff[:,0] = diff[:,0]*2
    print(f'diff : {diff}')
    distances = np.linalg.norm(diff, axis=1)
    closest_index = np.argmin(distances)
    closest_coordinate = coordinates[closest_index]
    return closest_coordinate, closest_index

def find_xy_in_roi(roi_xy1,roi_xy2,xys, base = 'center'):
    print(f'roi_xy1 = {roi_xy1}')
    print(f'roi_xy2 = {roi_xy2}')
    print(f'xys = {xys}')
    if base=='center':
        origin = [int(sum(x)/2) for x in zip(roi_xy1,roi_xy2)]
    else :
        orgin = [0,0]
    print(f'origin = {origin}')
    xy_list = []
    for xy in xys:
        x = xy[0]
        y = xy[1]
        if roi_xy1[0] < x and roi_xy2[0] > x:
            if roi_xy1[1] < y and roi_xy2[1] > y:
                xy_list.append(xy)
    if len(xy_list)!=0: 
        result, result_ind = find_closest_coordinate(xy_list,origin)
        #print(f'yx = {result} (index)={result_ind}')
        return result,result_ind
    else:
        return None,None

def gen_result(base_path = BASE_PATH):
 
    start_time = time.time()

    # PNG 파일을 저장할 리스트
    img_files = []
    # 디렉토리 순회
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 파일 확장자가 '.png' 인지 확인
            #if file.endswith('.png'):
                # 파일의 전체 경로를 생성하여 리스트에 추가
            file_path = os.path.join(root, file)
            img_files.append(file_path)

    # 파일 이름 순으로 정렬
    img_files.sort()
    
    execution_time = time.time() - start_time
    print(f"- - - - - 1: {execution_time}초")
    start_time = time.time()

    results = YOLO_MODEL_COUNT.predict(source=img_files, save=True, save_txt=True, save_conf=True, conf=0.75, iou = 0.9, device='cuda:0')

    execution_time = time.time() - start_time
    print(f"- - - - - 2: {execution_time}초")  #0.9
    start_time = time.time()
    
    # Create SAMPredictor
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="/home/jeonggyu/rapid_kit_detection/models/sam/sam_b.pt", save=True, device="cuda:1")
    #overrides = dict(max_det=10, half=True, conf=0.25, task='segment', mode='predict', imgsz=256, model="/home/jeonggyu/rapid_kit_detection/models/sam/sam_b.pt", save=True, device="cuda:1")
    #overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="/home/jeonggyu/rapid_kit_detection/models/sam/mobile_sam.pt", save=True,device="cuda:1")
    predictor = SAMPredictor(overrides=overrides)

    FSAM = FastSAM('FastSAM-x.pt')

    execution_time = time.time() - start_time
    print(f"- - - - - 3: {execution_time}초")
    start_time = time.time()

    #print(len(results))
    for result in results: #yolo 결과들 중 한 이미지에 대한 결과
    #for result in results[5:7]: #yolo 결과들 중 한 이미지에 대한 결과
        print(f'path : {result.path}')
        #print(f'masks : {(result.boxes is not None)}')
        if result.boxes is not None:
            start_time = time.time()
            
            #이미지에서 kit들 찾아내고 각 kit들 transform한 뒤, 각 kit에서 strip 추출 후 결과 저장
            image_rgb, cropped_kits = crop_image_from_yolo_result(result,target_class=0)#for kit
            print(f'cropped kits (kit): {len(cropped_kits)}')
            t_cropped_strip_dict_list = []
            #transformed_kit_list = []
            if len(cropped_kits) == 0:
                return 0
            
            execution_time = time.time() - start_time
            print(f"- - - - - 4: {execution_time}초")
            
            for i in range(len(cropped_kits)):
                start_time = time.time()
                
                kit_width = cropped_kits[i]['img'].shape[0]
                kit_heigth = cropped_kits[i]['img'].shape[1]
                kit_center = cropped_kits[i]['center']
                kit_origin = [kit_center[0]-int(kit_heigth/2),kit_center[1]-int(kit_width/2)]
                transformed_kit = perspective_transform_v3(cropped_kits[i],predictor, result.orig_img)
                #transformed_kit = perspective_transform_v4(cropped_kits[i],FSAM, result.orig_img)
                #transformed_kit = perspective_transform_v5(cropped_kits[i], result.orig_img)
                #transformed_kit_list.append(transformed_kit)

                execution_time = time.time() - start_time
                print(f"- - - - - 5-1: {execution_time}초")
                start_time = time.time()

                kit_yolo_result = YOLO_MODEL.predict(source=transformed_kit, save=True, save_txt=True, save_conf=True,  conf=0.1, iou = 0.9,classes=[1], project='strip_outputs')
                #print(kit_yolo_result)
                _ , t_cropped_strips = crop_image_from_yolo_result(kit_yolo_result[0],target_class=1,bgr=False)#for strip
                
                execution_time = time.time() - start_time
                print(f"- - - - - 5-2: {execution_time}초")
                start_time = time.time()
                
                if len(t_cropped_strips) != 0:
                    print(f'len(t_cropped_strips) = {len(t_cropped_strips)}')
                    
                    #가장 strip에 가까운 이미지 선택 
                    if len(t_cropped_strips)>1:
                        kit_width_t = transformed_kit.shape[0]
                        kit_heigth_t = transformed_kit.shape[1]
                        kit_center_t = [kit_width_t//2,kit_heigth_t//2]
                        print(f'kit_center_t={kit_center_t}')
                        kit_origin_t = [kit_center_t[0]-kit_width_t//2,kit_center_t[1]-kit_heigth_t//2]
                        xys = [ t_cropped_strips[ii]['center'] for ii in range(len(t_cropped_strips))]
                        #t_center,t_center_ind = find_xy_in_roi(kit_origin_t,[kit_center_t[0]+kit_width_t//2,kit_center_t[1]+kit_heigth_t//2],xys)
                        t_center,t_center_ind = find_xy_in_roi(kit_origin_t,[kit_heigth_t,kit_width_t],xys)
                    else :
                        t_center = t_cropped_strips[0]['center']
                        t_center_ind=0
                    print('t_cropped_strips center = %s,%s'%(t_center,t_center_ind))
                    t_cropped_strip_dict_list.append({'img':t_cropped_strips[t_center_ind]['img'],
                                                    'center':t_center,
                                                    'transformed_kit_img':transformed_kit,
                                                    'kit_width':kit_width,
                                                    'kit_heigth':kit_heigth,
                                                    'kit_origin':kit_origin})

                else:
                    print('strip can not found')
                
                execution_time = time.time() - start_time
                print(f"- - - - - 5-3: {execution_time}초")
                start_time = time.time()

            start_time = time.time()

            result_image_gen_v3(image_rgb, t_cropped_strip_dict_list, result.path, figsize=20)
            
            execution_time = time.time() - start_time
            print(f"- - - - - 6: {execution_time}초")

def count_kit(base_path = BASE_PATH):
    # PNG 파일을 저장할 리스트
    img_files = []

    # 디렉토리 순회
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 파일 확장자가 '.png' 인지 확인
            #if file.endswith('.png'):
                # 파일의 전체 경로를 생성하여 리스트에 추가
            file_path = os.path.join(root, file)
            img_files.append(file_path)

    # 파일 이름 순으로 정렬
    img_files.sort()

    results = YOLO_MODEL_COUNT.predict(source=img_files, save=True, save_txt=False, save_conf=True, conf=0.75, iou = 0.9, project='count_outputs',classes=[0])

    cnt_list = []
    for result in results: #yolo 결과들 중 한 이미지에 대한 결과
        cnt = 0
        if result.boxes is None:
            return cnt_list
        for box in result.boxes:
            if box.cls==2:
                cnt += 1 
        cnt_list.append(cnt)

    return cnt_list
    
