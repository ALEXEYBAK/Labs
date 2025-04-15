import numpy as np
from PIL import  Image, ImageOps
import CompGraphics as gc
import random
#Моделька
f = open("model_1.obj")
tz = 0.1
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
    tx, ty = 0, -0.05
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

angle_x = -15
angle_y = 215
angle_z = 0
vectorv = rotate_image(vectorv,angle_x,angle_y,angle_z)

def bar_coordinates(proj_x, proj_y, proj_x0, proj_y0, proj_x1, proj_y1, proj_x2, proj_y2):
    denom = ((proj_x0 - proj_x2) * (proj_y1 - proj_y2) - (proj_x1 - proj_x2) * (proj_y0 - proj_y2))
    if abs(denom) < 1e-6:
        return -1, -1, -1  # Вернуть значения, указывающие на то, что точка вне треугольника

    lambda0 = ((proj_x - proj_x2) * (proj_y1 - proj_y2) - (proj_x1 - proj_x2) * (proj_y - proj_y2)) / denom
    lambda1 = ((proj_x0 - proj_x2) * (proj_y - proj_y2) - (proj_x - proj_x2) * (proj_y0 - proj_y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def rendering(x1,y1 ,z1,x2,y2,z2,x3,y3,z3,I_0,I_1,I_2):
    
    height, width = 2000,2000
    ax = 10000
    a = ax*tz
    proj_x1 = x1*a/z1 + 1000;
    proj_y1 = y1*a/z1 + 1000;
    
    proj_x2 = x2*a/z2 + 1000;
    proj_y2 = y2*a/z2 + 1000;
    
    proj_x3 = x3*a/z3 + 1000;
    proj_y3 = y3*a/z3 + 1000;
    
    # Определение ограничивающего прямоугольника
    xmin = max(0, int(min(proj_x3, proj_x1, proj_x2)))
    ymin = max(0, int(min(proj_y3, proj_y1, proj_y2)))
    xmax = min(width, int(max(proj_x3, proj_x1, proj_x2))+1)
    ymax = min(height, int(max(proj_y3, proj_y1, proj_y2))+1)
    # color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    
    if(((proj_x1 - proj_x3) * (proj_y2 - proj_y3) - (proj_x2 - proj_x3) * (proj_y1 - proj_y3))==0.0):
        return
    else:
        # color = (-255*cos,-255*cos,-255*cos)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                alpha, beta, gamma = bar_coordinates(x, y,proj_x1,proj_y1,proj_x2,proj_y2,proj_x3,proj_y3)
                if ((alpha >= 0) and (beta >= 0) and (gamma >= 0)):
                    if (alpha*z1 + beta*z2 +gamma*z3) < z_buffer[y, x]:
                        color = -225*(alpha*I_0 + beta*I_1 + gamma*I_2)
                        # print(color);
                        z_buffer[y, x] = alpha*z1 + beta*z2 +gamma*z3
                        if (color>0):
                            img_mat2[y, x] = color

normals = np.zeros((len(vectorv), 3))
def normal(proj_x0,proj_y0,proj_z0,proj_x1,proj_y1,proj_z1,proj_x2,proj_y2,proj_z2,face_index):
    v1=np.array([proj_x1-proj_x2,proj_y1-proj_y2,proj_z1-proj_z2])
    v2=-np.array([proj_x0-proj_x1,proj_y0-proj_y1,proj_z0-proj_z1])
    n = np.cross(v1,v2)
    n_temp = n
    
    normals[vectorf[face_index][0] - 1] += n_temp
    
    normals[vectorf[face_index][1] - 1] += n_temp
    
    
    normals[vectorf[face_index][2] - 1] += n_temp
    
    return n

z_buffer = np.full((2000, 2000), float('inf'))  # Инициализация z-буфера

for i in range(0,len(vectorf)):
    x0 = vectorv[vectorf[i][0]-1][0]
    y0 = vectorv[vectorf[i][0]-1][1]
    z0 = vectorv[vectorf[i][0]-1][2]
    x1 = vectorv[vectorf[i][1]-1][0]
    y1 = vectorv[vectorf[i][1]-1][1]
    z1 = vectorv[vectorf[i][1]-1][2]
    x2 = vectorv[vectorf[i][2]-1][0]
    y2 = vectorv[vectorf[i][2]-1][1]
    z2 = vectorv[vectorf[i][2]-1][2]
    n = normal(x0, y0, z0, x1, y1, z1,x2, y2, z2, i)

for i in range(0,len(normals)):
    normals[i] = normals[i]/np.linalg.norm(normals[i])

for i in range(0,len(vectorf)):
    x0 = vectorv[vectorf[i][0]-1][0]
    y0 = vectorv[vectorf[i][0]-1][1]
    z0 = vectorv[vectorf[i][0]-1][2]
    x1 = vectorv[vectorf[i][1]-1][0]
    y1 = vectorv[vectorf[i][1]-1][1]
    z1 = vectorv[vectorf[i][1]-1][2]
    x2 = vectorv[vectorf[i][2]-1][0]
    y2 = vectorv[vectorf[i][2]-1][1]
    z2 = vectorv[vectorf[i][2]-1][2]
    l = [0, 0, 1]
    v1=np.array([x1-x2,y1-y2,z1-z2])
    v2=-np.array([x0-x1,y0-y1,z0-z1])
    n = np.cross(v1,v2)
    norm_n = np.linalg.norm(n)
    norm_l = np.linalg.norm(l)
    cos = np.dot(n, l) / (norm_n * norm_l)
    
    I_0 = normals[vectorf[i][0] - 1][2]
    I_1 = normals[vectorf[i][1] - 1][2]
    I_2 = normals[vectorf[i][2] - 1][2]
    if(cos<0):
        rendering(x0 , y0 , z0 ,
                x1 , y1 , z1,
                x2 , y2 , z2,
                I_0,I_1,I_2)

# for fl in vectorv:
#     img_mat2[round(fl[1]*5000) + 250, round(fl[0]*5000)+ 500] = (255, 255, 255)
img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img1.jpg")
