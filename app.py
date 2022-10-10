# TODO 
# Предусмотреть сбой 
# - если прилетела битая картинка камера не детектится 
# - если пустой файл 

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import cosine_similarity

import pickle 
import numpy as np 
from PIL import Image


# Модель извлечения эмбедингов изображений для детекции камеры 
resnet18 = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
feature_extractor.eval()

# Эмединги камер список из пары (id камеры, вектор камеры) для каждой камеры по 5 векторов 
with open('camera_vectors.p', 'rb') as f:
    camera_vectors = pickle.load(f)

# Подготовка картинок для извлечения эмбедингов     
tr = transforms.Compose([ 
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                    
                    ])

# модель для детексции людей на изображении  
model_bbox = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_bbox.classes = [0] # ищем только людей   

# Маски опасной зоны 
with open('line_masks.p' , 'rb') as f: 
    line_masks = pickle.load(f)

# поиск ближайших по косинусному расстоянию векторов камер 
def top_sim_image(v,vectors):

    sims = cosine_similarity(np.expand_dims(v, 0), vectors)

    sims = list(np.squeeze(sims))

    relation_dict = {i:s for i,s in zip(list(range(len(vectors))),sims)}

    out = sorted(relation_dict.items(), key = lambda x: x[1], reverse = True)

    return out 

  
def detect_camera(img):
    
    img_tensor = tr(img).unsqueeze(0)
    
    img_vector = feature_extractor(img_tensor).view(-1).detach().cpu().numpy()     

    vectors = np.array([i[1] for i in camera_vectors])
        
    out = top_sim_image(img_vector,vectors)
    
    camera_id = camera_vectors[out[0][0]][0] # 
                       
    return camera_id

def extract_bbox(file):
    
    #bboxs = []
    
    result = model_bbox(file)
    
    r = result.pandas().xyxy[0]
    
    r = r[['xmin','ymin','xmax','ymax']]
    
    bboxs = r.values.tolist()
    
    bboxs = [i for i in bboxs if len(i) > 0]

    #print('len(bboxs)',len(bboxs) )
    
    return bboxs

def detect_voilation(bbox, line_mask_data): 

    #print(bbox)
    
    x1, y1, x2, y2 = bbox
    
    #print('y2',y2)
    
    
    y2 = round(y2) -1
    
    line_mask = line_mask_data['mask']
    
    #print('line_mask',line_mask.shape)
    
    left_line = line_mask_data['left_line']
    
    delta = 10000 # todo
    
    if left_line:
        # линия слева 
        
        x_lim = [n for n,i in enumerate(im_mask[round(y2)]) if i]
        
        if len(x_lim) > 0:
            x_lim = x_lim[0]
            delta = round(x1) - x_lim   
        
    else:
        # линия справа
        
        x_lim = [n for n,i in enumerate(line_mask[y2]) if i]
        
        if len(x_lim) > 0:
            x_lim = x_lim[0]
            delta = x_lim - round(x2)   
    # 
    tr = 10 
    if delta < tr: #
        return True  
    else: 
        return False 
  
    
def predict(file): # file - > Название файла с картинкой в директории images 
    
    img = Image.open(file)   
        
    camera_id = detect_camera(img)                  
    
    line_mask_data = line_masks[camera_id] 
    
    out = [] 
    
    for bbox in extract_bbox(file):
        
        if detect_voilation(bbox, line_mask_data):
            out.append({'violation':True,'file':file,'camera_id':camera_id,'bbox': bbox})            
        else:
            out.append({'violation':False,'file':file,'camera_id':camera_id,'bbox': bbox})  
            
    return out  
    









