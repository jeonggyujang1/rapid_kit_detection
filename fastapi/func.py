import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import scipy as sc
from ultralytics.models.sam import Predictor as SAMPredictor

from ultralyticsplus import YOLO
import matplotlib.patches as patches
from scipy.signal import find_peaks, savgol_filter

# HoughLines threshold
HOUGHLINE_RHO=0.5
HOUGHLINE_THR=100
BBOX_MARGIN=0
# 특정 경로
BASE_PATH = '/datalakes/0000/rapid_kit_custom/test/images'


def crop_image_from_yolo_result(result, target_class=3, bgr=True):
    if bgr:
        image_bgr = result.orig_img
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = result.orig_img
    boxes = result.boxes

    cropped_imgs = list()
    for box in boxes:
        if box.cls==target_class:
            x1,y1,x2,y2 = box.xyxy.int().tolist()[0]
            x1,y1,x2,y2 = x1-BBOX_MARGIN,y1-BBOX_MARGIN,x2+BBOX_MARGIN,y2+BBOX_MARGIN
            #print(x1,y1,x2,y2)
            cropped_imgs.append({'img': image_rgb[y1:y2,x1:x2], 'center': [int((x1+x2)/2),int((y1+y2)/2)]})

    cropped_imgs = sorted(cropped_imgs, key=lambda xy: xy['center'][0]+xy['center'][1])  # x 좌표를 기준으로 정렬
    return image_rgb, cropped_imgs

# 이미지를 그레이스케일로 변환
def rotation_kit(input_img):
    if input_img.shape[0]>=input_img.shape[1]:
        orient = 'vertical'
    else:
        orient = 'horizon'
    
    img = input_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny 에지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 허프 변환을 이용하여 직선 검출
    lines = cv2.HoughLines(edges, HOUGHLINE_RHO, np.pi / 180, HOUGHLINE_THR)
    #print(lines)
    if lines is None:
        rotated_img=img
        return rotated_img
    
    lines_ = list()
    for line in lines:
        if np.abs(line[0][0])>5:
            lines_.append(line)
    lines = lines_
    
    if len(lines)==0:
        rotated_img=img
        return rotated_img
    # 검출된 직선들을 반복하여 그리기
    '''
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''
    # 검출된 직선들의 각도 계산
    angles = [line[0][1] for line in lines]
    #print(angles)
    # 직선들의 각도 중앙값 계산
    median_angle = np.median(angles)
    # 회전 각도 계산
    rotation_angle = np.degrees(median_angle)
    print(orient,rotation_angle,end=' -> ')

    if orient=='vertical':
        if rotation_angle>90:
            rotation_angle = rotation_angle-180
    elif orient=='horizon':
        rotation_angle = rotation_angle-90
        if rotation_angle>90:
            rotation_angle = rotation_angle-180
        
    print(rotation_angle, " - ", len(lines) )
    
    # 이미지 중심 좌표 계산
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # 회전 변환 행렬 계산
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 이미지 회전
    rotated_img = cv2.warpAffine(img, M, (w, h),borderValue=(255,255,255))
    # 결과 이미지 출력
    return rotated_img 

def c_value_line_from_img(img, strip_crop_ratio=1):

    # 이미지의 높이와 너비를 가져옵니다.
    height, width = img.shape[:2]

    # crop할 영역의 높이와 너비를 계산합니다.
    crop_height = int(height * strip_crop_ratio)
    crop_width = int(width * strip_crop_ratio)

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

def perspective_transform(input_img,predictor):
    # Set image
    predictor.set_image(input_img)  # set with image file
    #predictor.set_image(cv2.imread(cropped_img))  # set with np.ndarray
    sam_results = predictor()

    # Reset image
    predictor.reset_image()

    xy = sam_results[0].masks[0].xy[0]
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

    #중앙으로부터 각 사분면에서 가장 먼 점

    edge_4pt = [[0,0],[0,0],[0,0],[0,0]]
    edge_4pt_dist = [0,0,0,0]
    for xy in rotated_points:
        cur_dist = math.sqrt((xy[0])**2 + (xy[1])**2)
        if xy[0] > 0 and xy[1] > 0:
            if edge_4pt_dist[0] < cur_dist:
                edge_4pt_dist[0] = cur_dist
                edge_4pt[0] = [xy[0],xy[1]]
        elif xy[0] < 0 and xy[1] > 0:
            if edge_4pt_dist[1] < cur_dist:
                edge_4pt_dist[1] = cur_dist
                edge_4pt[1] = [xy[0],xy[1]]
        elif xy[0] < 0 and xy[1] < 0:
            if edge_4pt_dist[2] < cur_dist:
                edge_4pt_dist[2] = cur_dist
                edge_4pt[2] = [xy[0],xy[1]]
        elif xy[0] > 0 and xy[1] < 0:
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

def result_image_gen_v2(image_rgb, cropped_imgs, image_path, figsize=20):
    rotated_imgs = list()
    value_lines = list()
    colors_list=list()
        
    for i in range(len(cropped_imgs)):
        rotated_img = rotation_kit(cropped_imgs[i]['img'])
        value_line, colors = c_value_line_from_img(rotated_img)
        rotated_imgs.append(rotated_img)
        value_lines.append(value_line)
        colors_list.append(colors)
    
    plt.figure(figsize=(figsize,figsize))

    
   # added_img = image_rgb+cropped_imgs[0]
    #plt.imshow(added_img)
    

    implot = plt.imshow(image_rgb)

    for img_ind in range(len(cropped_imgs)):
        strip_width = np.array(cropped_imgs[img_ind]['img'].shape[:2]).min()#*0.9
        strip_heigth = np.array(cropped_imgs[img_ind]['img'].shape[:2]).max()#*0.9
        strip_center = cropped_imgs[img_ind]['center']
        if cropped_imgs[img_ind]['img'].shape[0]>cropped_imgs[img_ind]['img'].shape[1]:
            x = np.linspace(0,1,len(value_lines[img_ind]))
            x = (x-np.min(x))/(np.max(x)-np.min(x))*(strip_width*3)
            x_offset = strip_center[0]-(strip_width*3)/2 #offset
            x = x+x_offset

            y_ = value_lines[img_ind].sum(axis=1)
            y = (y_-np.min(y_))/(np.max(y_)-np.min(y_))*(strip_heigth*0.6)
            y_offset = strip_center[1]-strip_heigth*1.5
            y = y+y_offset
            
            #bounding box
            rect = patches.Rectangle((strip_center[0]-strip_width/2, strip_center[1]-strip_heigth/2), strip_width, strip_heigth, linewidth=0.5, edgecolor='g', facecolor='None')
            plt.gca().add_patch(rect)
            
            rotation = 0
            
        else:
            x = np.linspace(0,1,len(value_lines[img_ind]))
            x = (x-np.min(x))/(np.max(x)-np.min(x))*(strip_width*3)
            x_offset =strip_center[0]-strip_heigth*1.5 #offset
            x = x+x_offset

            y_ = value_lines[img_ind].sum(axis=1)
            y = (y_-np.min(y_))/(np.max(y_)-np.min(y_))*(strip_heigth*0.6)
            y_offset = strip_center[1]-(strip_width*3)/2
            y = y+y_offset
            
            #bounding box
            rect = patches.Rectangle((strip_center[0]-strip_heigth/2, strip_center[1]-strip_width/2), strip_heigth, strip_width, linewidth=0.5, edgecolor='g', facecolor='None')
            plt.gca().add_patch(rect)
            
            rotation = 1

        #plt.scatter(x,y,c=colors_list[img_ind],s=figsize//5)
        #plt.scatter(strip_center[0],strip_center[1],c='red',s=10)

        rect = patches.Rectangle((np.min(x)-20, np.min(y)-20), (np.max(x) - np.min(x)+40), (np.max(y) - np.min(y)+40), linewidth=0.5, edgecolor='k', facecolor='w')
        plt.gca().add_patch(rect)

        plt.scatter(x,y,c=colors_list[img_ind],s=figsize//5)
        
        y_t = -y_/3+255
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
        #print(peaks)
        background_x = list()
        background_y = list()
        for peak in peaks:
            if rotation==0:
                x_peak = strip_center[0] - strip_width//2 + strip_width*1.1
                y_peak = strip_center[1] - strip_heigth//2 + peak
                plt.annotate(f"{smoothed[peak]:.0f}", (x_peak, y_peak), c='k', fontsize=figsize//2)
                background_x.append(strip_center[0])
                background_y.append(strip_center[1] - strip_heigth//2 + peak)
            else:
                x_peak = strip_center[0] - strip_heigth//2 + peak - figsize
                y_peak = strip_center[1] - strip_width//2 + strip_width*1.5
                plt.annotate(f"{smoothed[peak]:.0f}", (x_peak, y_peak), c='k', fontsize=figsize//2)
                background_x.append(strip_center[0] - strip_heigth//2 + peak)
                background_y.append(strip_center[1])
        x_peak = int(sum(background_x)/len(background_x))
        y_peak = int(sum(background_y)/len(background_y))
        #print(background_x,background_y)
        #print(x_peak,y_peak)
        #print('\n')
        if rotation==0:
            x_peak_= x_peak - strip_width//2 - strip_width*0.3
            y_peak_= y_peak
            plt.annotate(f"{smoothed[y_peak-strip_center[1]]:.0f}", (x_peak_, y_peak_), c='k', fontsize=figsize//2)
        else:
            x_peak_= x_peak
            y_peak_= y_peak - strip_width//2 - strip_width*0.1
            plt.annotate(f"{smoothed[x_peak-strip_center[0]]:.0f}", (x_peak_, y_peak_), c='k', fontsize=figsize//2)

    #return y_
    plt.show()
    #file_name = image_path.split('/')[-1]
    #plt.savefig(f'./analysis_results/{file_name}')
    #print(f'./analysis_results/{file_name} (saved)')

def result_image_gen_v3(image_rgb, cropped_imgs, image_path, figsize=10):
    value_lines = list()
    colors_list=list()
    fontsize = figsize//2+5   
    strip_crop_ratio = 0.75
    
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
                x_peak = strip_center[0]
                y_peak = peak
                #print(rotation, x_peak,y_peak)
                plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
                rect = patches.Arrow(x_peak-(strip_width*strip_crop_ratio)//2,y_peak,strip_width*strip_crop_ratio,0 ,alpha = 0.3, linewidth=0.5, edgecolor='magenta', facecolor='None')
                plt.gca().add_patch(rect)
                plt.annotate(f"{smoothed[peak]:.0f}", (x_peak, y_peak), c='magenta', fontsize=fontsize)
                #background_x.append(strip_center[0])
                #background_y.append(strip_center[1] - strip_heigth//2 + peak)
            else:
                x_peak = peak - 5
                y_peak = strip_center[1] - strip_width//2 + peak_ind*10 
                #print(rotation, x_peak,y_peak)
                plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
                rect = patches.Arrow(x_peak,y_peak-(strip_width*strip_crop_ratio)//2,0,strip_width*strip_crop_ratio ,alpha = 0.3, linewidth=0.5, edgecolor='magenta', facecolor='None')
                plt.gca().add_patch(rect)
                plt.annotate(f"{smoothed[peak]:.0f}", (x_peak, y_peak), c='magenta', fontsize=fontsize)
                #background_x.append(strip_center[0] - strip_heigth//2 + peak)
                #background_y.append(strip_center[1])
        #x_peak_ = int(sum(background_x)/len(background_x))
        #y_peak_ = int(sum(background_y)/len(background_y))
        x_peak_ = strip_center[0]
        y_peak_ = strip_center[1]
        #print(background_x,background_y)
        #print(x_peak,y_peak)
        #print('\n')
        #print(strip_width)
        if rotation==0:
            plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
            plt.annotate(f"{smoothed[y_peak_]:.0f}", (x_peak_-strip_width//3, y_peak_), c='green', fontsize=fontsize)
            rect = patches.Arrow(x_peak_-(strip_width*strip_crop_ratio)//2,y_peak_,strip_width*strip_crop_ratio,0 ,alpha = 0.3, linewidth=0.5, edgecolor='green', facecolor='None')
            plt.gca().add_patch(rect)
        else:
            plt.subplot(subplot_x, subplot_y, subplot_x_kit*subplot_y + 2*img_ind+1)
            plt.annotate(f"{smoothed[x_peak_]:.0f}", (x_peak_, y_peak_-strip_width//3), c='green', fontsize=fontsize)
            rect = patches.Arrow(x_peak_,y_peak_-(strip_width*strip_crop_ratio)//2,0,strip_width*strip_crop_ratio ,alpha = 0.3, linewidth=0.5, edgecolor='green', facecolor='None')
            plt.gca().add_patch(rect)
    #return y_
    plt.subplots_adjust(wspace=0,hspace=0.1)
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

    # 결과 출력
    #print(img_files)

    model = YOLO('/home/jeonggyu/rapid_kit_detection/best.pt')
    results = model.predict(source=img_files, save=True, save_txt=False, conf=0.7)

    # Create SAMPredictor
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="/home/jeonggyu/rapid_kit_detection/sam_b.pt", save=True)
    predictor = SAMPredictor(overrides=overrides)

    #print(len(results))
    for result in results: #yolo 결과들 중 한 이미지에 대한 결과
    #for result in results[5:7]: #yolo 결과들 중 한 이미지에 대한 결과
        print(f'path : {result.path}')
        #print(f'masks : {(result.boxes is not None)}')
        if result.boxes is not None:
            #이미지에서 strip들 찾아내고 각 strip center 값 별도 저장
            image_rgb, cropped_strips = crop_image_from_yolo_result(result,target_class=3)#for strip
            strip_center_list = []
            for i in range(len(cropped_strips)): 
                strip_center_list.append(cropped_strips[i]['center'])
            #print(strip_center_list)
            
            #이미지에서 kit들 찾아내고 각 kit들 transform한 뒤, 각 kit에서 strip 추출 후 결과 저장
            image_rgb, cropped_kits = crop_image_from_yolo_result(result,target_class=2)#for kit
            print(f'cropped kits (kit): {len(cropped_kits)}')
            t_cropped_strip_dict_list = []
            transformed_kit_list = []
            if len(cropped_kits) == 0:
                return 0
            for i in range(len(cropped_kits)):
                kit_width = cropped_kits[i]['img'].shape[0]
                kit_heigth = cropped_kits[i]['img'].shape[1]
                kit_center = cropped_kits[i]['center']
                kit_origin = [kit_center[0]-int(kit_heigth/2),kit_center[1]-int(kit_width/2)]
                transformed_kit = perspective_transform(cropped_kits[i]['img'],predictor)
                transformed_kit_list.append(transformed_kit)
                #plt.imshow(transformed_kit)
                kit_yolo_result = model.predict(source=transformed_kit, save=True, save_txt=False, conf=0.5)
                #print(kit_yolo_result)
                _ , t_cropped_strips = crop_image_from_yolo_result(kit_yolo_result[0],target_class=3,bgr=False)#for strip
                '''
                center,center_ind = find_xy_in_roi(kit_origin,
                                        [kit_center[0]+int(kit_heigth/2),kit_center[1]+int(kit_width/2)],
                                        strip_center_list)
                '''
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
                    #v2
                    '''
                    t_cropped_strips[0]['center'][0] += kit_origin[0]
                    t_cropped_strips[0]['center'][1] += kit_origin[1]
                    print('t_cropped_strips center + kit_origin= %s'%(t_cropped_strips[0]['center']))
                    print(f't_cropped_strips = {len(t_cropped_strips)}')
                    t_cropped_strip_dict_list.append({'img':t_cropped_strips[0]['img'],'center':center})
                    '''
                    #v3
                    t_cropped_strip_dict_list.append({'img':t_cropped_strips[t_center_ind]['img'],
                                                    'center':t_center,
                                                    'transformed_kit_img':transformed_kit,
                                                    'kit_width':kit_width,
                                                    'kit_heigth':kit_heigth,
                                                    'kit_origin':kit_origin})

                else:
                    print('strip can not found')
            #transformed_imgs = (for strip) [{'img':--, 'center':--},{}...] 
            #result_image_gen_v2(image_rgb, t_cropped_strip_dict_list, result.path, figsize=20)
            result_image_gen_v3(image_rgb, t_cropped_strip_dict_list, result.path, figsize=20)
