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


def bar_coordinates(x,y,x0,y0,x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0,lambda1,lambda2

def rendering(x1,y1,z1,x2,y2,z2,x3,y3,z3,cos):
    height, width = 2000,2000
    # Определение ограничивающего прямоугольника
    xmin = max(0, int(min(x3, x1, x2)))
    ymin = max(0, int(min(y3, y1, y2)))
    xmax = min(width, int(max(x3, x1, x2))+1)
    ymax = min(height, int(max(y3, y1, y2))+1)
    # color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    
    if(((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))==0):
        return
    else:
        color = (-255*cos,-255*cos,-255*cos)
        for x in range(xmin, xmax):
            for y in range(ymin, ymax):
                alpha, beta, gamma = bar_coordinates(x, y,x1,y1,x2,y2,x3,y3)
                if ((alpha >= 0) and (beta >= 0) and (gamma >= 0)):
                    if (alpha*z0 + beta*z1 +gamma*z2) < z_buffer[y, x]: 
                        z_buffer[y, x] = alpha*z0 + beta*z1 +gamma*z2
                        img_mat2[y, x] = color


def normal(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    v1=np.array([x1-x2,y1-y2,z1-z2])
    v2=-np.array([x0-x1,y0-y1,z0-z1])
    return np.cross(v1,v2) 

def calc_svet(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    l = [0, 0, 1]
    n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    norm_n = np.linalg.norm(n)

    if norm_n < 1e-6: 
        return 0  

    norm_l = np.linalg.norm(l)
    return np.dot(n, l) / (norm_n * norm_l)

z_buffer = np.full((2000, 2000), float('inf'))  # Инициализация z-буфера
for i in range(0,len(vectorf)):
    x0 = int((vectorv[vectorf[i][0]-1][0])*10000 + 1000)
    y0 = int((vectorv[vectorf[i][0]-1][1])*10000 + 500)
    z0 = int((vectorv[vectorf[i][0]-1][2])*10000 +500)
    x1 = int((vectorv[vectorf[i][1]-1][0])*10000 + 1000)
    y1 = int((vectorv[vectorf[i][1]-1][1])*10000 + 500)
    z1 = int((vectorv[vectorf[i][1]-1][2])*10000+500)
    x2 = int((vectorv[vectorf[i][2]-1][0])*10000 + 1000)
    y2 = int((vectorv[vectorf[i][2]-1][1])*10000 + 500)
    z2 = int((vectorv[vectorf[i][2]-1][2])*10000+500)
    # gc.bresanham(img_mat2, x0, y0, x1, y1, color)
    # gc.bresanham(img_mat2, x1, y1, x2, y2, color)
    # gc.bresanham(img_mat2, x0, y0, x2, y2, color)
    cos = calc_svet(x0,y0,z0,x1,y1,z1,x2,y2,z2)
    if(cos<0):
        rendering(x0,y0,z0,x1,y1,z1,x2,y2,z2,cos)

# for fl in vectorv:
#     img_mat2[round(fl[1]*5000) + 250, round(fl[0]*5000)+ 500] = (255, 255, 255)
img = Image.fromarray(img_mat2, mode="RGB")
img = ImageOps.flip(img)
img.save("img.jpg")