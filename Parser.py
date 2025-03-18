import numpy as np
from PIL import  Image, ImageOps
import CompGraphics as gc
import random
#Моделька
f = open("model_1.obj")

vectorv = []
vectorf = []
for line in f:
    v = line.split()
    if (v[0] == "v"):
        vectorv.append([float(v[1]), float(v[2]), float(v[3])])
    if (v[0] == "f"):
        v1 = v[1].split('/')[0]
        v2 = v[2].split('/')[0]
        v3 = v[3].split('/')[0]
        vectorf.append([int(v1), int(v2), int(v3)])

#Соединить все вершины треугольника, нарисовать хранится в виде: (x1, x2, x3)(y1,y2,y3)(z1,z2,z3)
img_mat2 = np.zeros(shape=(2000, 2000, 3), dtype=np.uint8)
color = (255,255,255)

def rotate_image(vertices, angle_x_degrees, angle_y_degrees, angle_z_degrees):
    tx, ty, tz = 0, -0.05,0
    angle_x_radians = np.radians(angle_x_degrees)
    angle_y_radians = np.radians(angle_y_degrees)
    angle_z_radians = np.radians(angle_z_degrees)
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x_radians), -np.sin(angle_x_radians)],
        [0, np.sin(angle_x_radians), np.cos(angle_x_radians)]
    ])
    rotation_matrix_y = np.array([
        [np.cos(angle_y_radians), 0, np.sin(angle_y_radians)],
        [0, 1, 0],
        [-np.sin(angle_y_radians), 0, np.cos(angle_y_radians)]
    ])
    rotation_matrix_z = np.array([
        [np.cos(angle_z_radians), -np.sin(angle_z_radians), 0],
        [np.sin(angle_z_radians), np.cos(angle_z_radians), 0],
        [0, 0, 1]
    ])
    combined_rotation_matrix = np.dot(rotation_matrix_x, np.dot(rotation_matrix_y, rotation_matrix_z))
    rotated_vertices = []
    for vertex in vertices:
        vertex_np = np.array(vertex)  #Преобразуем вершину в массив NumPy
        rotated_vertex = np.dot(combined_rotation_matrix, vertex_np)  #Умножаем 
        rotated_vertex = rotated_vertex + np.array([tx, ty, tz]) 
        rotated_vertices.append(rotated_vertex.tolist())
    return rotated_vertices

angle_x = 0
angle_y = 200
angle_z = 0
vectorv = rotate_image(vectorv,angle_x,angle_y,angle_z)

def bar_coordinates(proj_x,proj_y,proj_x0,proj_y0,proj_x1,proj_y1,proj_x2,proj_y2):
    lambda0 = ((proj_x - proj_x2) * (proj_y1 - proj_y2) - (proj_x1 - proj_x2) * (proj_y - proj_y2)) / ((proj_x0 - proj_x2) * (proj_y1 - proj_y2) - (proj_x1 - proj_x2) * (proj_y0 - proj_y2))
    lambda1 = ((proj_x0 - proj_x2) * (proj_y - proj_y2) - (proj_x - proj_x2) * (proj_y0 - proj_y2)) / ((proj_x0 - proj_x2) * (proj_y1 - proj_y2) - (proj_x1 - proj_x2) * (proj_y0 - proj_y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0,lambda1,lambda2

def rendering(x1,y1 ,z1,x2,y2,z2,x3,y3,z3,cos):
    
    height, width = 2000,2000
    proj_x1 = x1*10000 + 1000;
    proj_y1 = y1*10000 + 1000;
    
    proj_x2 = x2*10000 + 1000;
    proj_y2 = y2*10000 + 1000;
    
    proj_x3 = x3*10000 + 1000;
    proj_y3 = y3*10000 + 1000;
    
    # Определение ограничивающего прямоугольника
    xmin = max(0, int(min(proj_x3, proj_x1, proj_x2)))
    ymin = max(0, int(min(proj_y3, proj_y1, proj_y2)))
    xmax = min(width, int(max(proj_x3, proj_x1, proj_x2))+1)
    ymax = min(height, int(max(proj_y3, proj_y1, proj_y2))+1)
    # color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    
    if(((proj_x1 - proj_x3) * (proj_y2 - proj_y3) - (proj_x2 - proj_x3) * (proj_y1 - proj_y3))==0):
        return
    else:
        color = (-255*cos,-255*cos,-255*cos)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                alpha, beta, gamma = bar_coordinates(x, y,proj_x1,proj_y1,proj_x2,proj_y2,proj_x3,proj_y3)
                if ((alpha >= 0) and (beta >= 0) and (gamma >= 0)):
                    if (alpha*z1 + beta*z2 +gamma*z3) < z_buffer[y, x]: 
                        z_buffer[y, x] = alpha*z1 + beta*z2 +gamma*z3
                        img_mat2[y, x] = color


def normal(proj_x0,proj_y0,proj_z0,proj_x1,proj_y1,proj_z1,proj_x2,proj_y2,proj_z2):
    v1=np.array([proj_x1-proj_x2,proj_y1-proj_y2,proj_z1-proj_z2])
    v2=-np.array([proj_x0-proj_x1,proj_y0-proj_y1,proj_z0-proj_z1])
    return np.cross(v1,v2) 

def calc_svet(proj_x0, proj_y0, proj_z0, proj_x1, proj_y1, proj_z1, proj_x2, proj_y2, proj_z2):
    l = [0, 0, 1]
    n = normal(proj_x0, proj_y0, proj_z0, proj_x1, proj_y1, proj_z1, proj_x2, proj_y2, proj_z2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-6: 
        return 0  
    norm_l = np.linalg.norm(l)
    return np.dot(n, l) / (norm_n * norm_l)

z_buffer = np.full((2000, 2000), float('inf'))  # Инициализация z-буфера
for i in range(0,len(vectorf)):
    x0 = (vectorv[vectorf[i][0]-1][0])
    y0 = (vectorv[vectorf[i][0]-1][1])
    z0 = (vectorv[vectorf[i][0]-1][2])
    x1 = (vectorv[vectorf[i][1]-1][0])
    y1 = (vectorv[vectorf[i][1]-1][1])
    z1 = (vectorv[vectorf[i][1]-1][2])
    x2 = (vectorv[vectorf[i][2]-1][0])
    y2 = (vectorv[vectorf[i][2]-1][1])
    z2 = (vectorv[vectorf[i][2]-1][2])
    # gc.bresanham(img_mat2, x0, y0, x1, y1, color)
    # gc.bresanham(img_mat2, x1, y1, x2, y2, color)
    # gc.bresanham(img_mat2, x0, y0, x2, y2, color)
    cos = calc_svet(x0,y0,z0,x1,y1,z1,x2,y2,z2)
    if(cos<0):
        rendering(x0 , y0 , z0 ,
                x1 , y1 , z1,
                x2 , y2 , z2,
                cos)




# for fl in vectorv:
#     img_mat2[round(fl[1]*5000) + 250, round(fl[0]*5000)+ 500] = (255, 255, 255)
img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img.jpg")
