# ###############FUNCTIONS################
import os, sys
import numpy as np
import gdal
import math
import itertools

#import functions
from qgis.core import QgsProject, QgsVectorFileWriter, QgsCoordinateReferenceSystem
from osgeo import ogr, osr
from qgis.core import QgsWkbTypes, QgsVectorLayer, QgsPointXY, QgsFields, QgsField, QgsFeature, QgsGeometry, QgsProject
from PIL import Image 
from numpy.lib import recfunctions as rfn
from qgis.utils import iface

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import pyqtSlot, QVariant
from PyQt5.QtWidgets import QLineEdit, QFileDialog, QTableWidgetItem, QMessageBox


#open raster file
#"raster_file_path" - path to the raster (grid) file 
def open_raster(self,raster_file_path):
    try:
        raster_file = gdal.Open(raster_file_path)
        return raster_file
    except IOError:
        QMessageBox(QMessageBox.Warning, "Error","Failed to open raster dataset.", QMessageBox.Ok).exec()

#create an array with a height from the raster
def raster_height_array(self,raster_file):
    #take information about a height for every pixel from the raster dataset
    raster_height_arr = np.array(raster_file.GetRasterBand(1).ReadAsArray())
    return (raster_height_arr)


#describe raster parameters
#"x_size", "y_size" - pixel size[m]
#"array_pix_size" - information about shape of an array
#"number_xpix", "number_ypix" - number of columns and rows[pix]
#geotran - describe raster position, pixel resolution: "x_left_corner", "y_left_corner", "pixel_width", "pixel_height"
def raster_parameters(self,raster_file,raster_height_arr):
    
    x_size = raster_file.RasterXSize
    y_size = raster_file.RasterYSize
    array_pix_size = raster_height_arr.shape
    number_ypix = int(array_pix_size[0])
    number_xpix = int(array_pix_size[1])
    geotran = raster_file.GetGeoTransform()
    x_left_corner = geotran[0]
    y_left_corner = geotran[3]
    pixel_width = geotran[1]
    pixel_height = geotran[5]
    prj = raster_file.GetProjection()
    srs = osr.SpatialReference(prj)
    if srs.IsProjected:
        EPSG_num = srs.GetAttrValue('PROJCS|AUTHORITY',1)
    else:
        print('Cannot find projection of the raster dataset.')

    raster_param_arr = np.array([x_size, y_size, array_pix_size, number_xpix, number_ypix, x_left_corner, y_left_corner, pixel_width, pixel_height,prj,srs,EPSG_num])
    return raster_param_arr


#create an array with the information for every pixel from the raster [xmin, xmax, ymin, ymax, z]
#"number_xpix", "number_ypix", "x_left_corner", "y_left_corner", "pixel_width", "pixel_height" - raster parameters 
#"raster_height_arr" - array with a height from the raster
def raster_XYZ_array(self, number_xpix, number_ypix, x_left_corner, y_left_corner, pixel_width, pixel_height, raster_height_arr):

    XYZ = np.array(number_ypix*[number_xpix*[5*[0]]],dtype='f')

    #pass through each pixel of the raster, first by rows and then by columns, save information about every pixel
    for i in range(0, number_ypix):
            for j in range(0, number_xpix):
                
                x_min = x_left_corner + j*pixel_width
                x_max = x_left_corner + (j+1)*pixel_width
                y_min = y_left_corner + (i+1)*pixel_height
                y_max = y_left_corner + i*pixel_height
                z = raster_height_arr[i,j]
                
                XYZ[i][j][0] = x_min
                XYZ[i][j][1] = x_max
                XYZ[i][j][2] = y_min
                XYZ[i][j][3] = y_max
                XYZ[i][j][4] = z
                
    return XYZ
    
    
#open the vector file with a linear layer
def open_vector(self,lines_file_path):
    try:
        lines = QgsVectorLayer(lines_file_path, 'szlaki2D', 'ogr')
        return lines
    except IOError:
        QMessageBox(QMessageBox.Warning, "Error","Failed to open linear layer.", QMessageBox.Ok).exec()


#return objects "fts" from the file with a linear layer - "lines"
def	vector_objects(self,lines):
    fts = lines.getFeatures()
    return fts


#return object "ft" as a polyline/multipolyline "elem_xy"
def vector_objects_elements(self,ft):
    geom_ft = ft.geometry()
    #check if the type is: single or not
    geom_type = QgsWkbTypes.isSingleType(geom_ft.wkbType())

    #check a detailed type of the geometry: polyline, multi-polyline or else
    if geom_ft.type() == QgsWkbTypes.LineGeometry:
        if geom_type:
            elem_xy = geom_ft.asPolyline()
        else:
            elem_xy = geom_ft.asMultiPolyline()
    else:
        QMessageBox(QMessageBox.Warning, "Error","Invalid linear layer format.", QMessageBox.Ok).exec()

    return elem_xy


#return an empty array with the optimal size
#"parameters_number" - last dimension of an array
def empty_array(self,fts,parameters_number):
    row_vert=0

    #-2 - it's only for the initialization, first dimension of an array should be connected with the objects numbers...
    #...second with the maximum numbers of vertices in any of the objects, last dimension should keep information about parameters
    col_arr = np.array([-2])

    for ft in fts:
        elem_xy = vector_objects_elements(self,ft)
        #adding numbers of vertices
        row_vert+=1
        
        for xy in elem_xy:
            num_vert = int(len(xy))
            #number of vertices in every objects
            col_arr = np.append(col_arr, num_vert)
    
    #choose maximum number of vertices and create an array
    col_vert = np.amax(col_arr)
    empty_vertex_array = np.array(row_vert*[col_vert*[parameters_number*[0]]], dtype='d')
    
    return empty_vertex_array


#return an array with [object_number, vertex_number, x, y] information for the linear layer
def vertex_information_array(self,lines,fts):
    
    vertex_array = empty_array(self,fts,4)
    
    for ft in lines.getFeatures():
        geom_ft = ft.geometry()
        object_number = ft.id()
        xy = vector_objects_elements(self,ft)
        
        #"num" - vertex number in a specified object
        num=0
        for v in xy:
            num_end = int(len(v))
            
            #for every vertices in objects read information about x,y and object number
            for num in range (0, num_end):
                vertex_ft = geom_ft.vertexAt(num)
                vertex_x = vertex_ft.x()
                vertex_y = vertex_ft.y()
                        
                vertex_array [object_number] [num] [0] = object_number
                vertex_array [object_number] [num] [1] = num
                vertex_array [object_number] [num] [2] = vertex_x
                vertex_array [object_number] [num] [3] = vertex_y
                
                num += 1
            
    return vertex_array


#return heights for points between different pixels
#"val1","val2","val3","val4" - height values for pixels around point, "val1" - checked pixel
def pixel_elevation(self,val1,val2,val3,val4):
    #help to calculate heights for points from pixels in a specific situation
    try:
        z = (val1 + val2 + val3 + val4)/4
    except:
        try:
            z = (val1 + val2)/2
        except:
            try:
                z = (val1 + val3)/2
            except:
                z = val1
    return z


#add height values to  points, from the raster into the specified array 
#i,j,k,l,n0,n1,n2,n3,n4,n5,n6 - numeric parameters
def add_elevation(self,XYZ,vertex_array,z_arr,k,l,n4,n5,n6,number_ypix,number_xpix,x_left_corner,y_left_corner,pixel_width,pixel_height):

    #check if a vertex is in the range of the raster area
    if (vertex_array [k][l][n5] < x_left_corner) or (vertex_array [k][l][n5] > x_left_corner+number_xpix*pixel_width) or (vertex_array [k][l][n6] > y_left_corner) or (vertex_array [k][l][n6] < y_left_corner+number_ypix*pixel_height):
        QMessageBox(QMessageBox.Warning, "Error","Invalid data range.", QMessageBox.Ok).exec()
    
    #j-columns, i -rows, the rest is calculated from the difference in the coordinates of the upper left pixel and the tested point of the linear layer
    j = (vertex_array [k][l][n5] - x_left_corner)%pixel_width
    i = (y_left_corner - vertex_array [k][l][n6])%pixel_height
    
    #if i, j different from 0, it means that the examined point lies in the *middle of the pixel and not on its edges
    if (i != 0) and (j != 0):
        #the position of the pixel is calculated and it gives the value of the height for the specified vertex 
        j = math.floor((vertex_array [k][l][n5] - x_left_corner)/pixel_width)
        i = math.floor((vertex_array [k][l][n6] - y_left_corner)/pixel_height)
        z = XYZ[i][j][n4]
        #"z_arr" - an array with information about heights
        z_arr = np.append(z_arr,z)
    
    else: 
        j = math.floor((vertex_array [k][l][n5] - x_left_corner)/pixel_width)
        i = math.floor((y_left_corner - vertex_array [k][l][n6])/pixel_height*(-1))
        
        #check *vertical pixel edges
        if (vertex_array [k][l][n5] - x_left_corner)%pixel_width == 0:
            if (y_left_corner - vertex_array [k][l][n6])%pixel_height != 0:
                
                if (vertex_array [k][l][n5] - x_left_corner) == 0:
                    z = XYZ[i][j][n4]
                elif (vertex_array [k][l][n5] - x_left_corner) == number_xpix*pixel_width:
                    z = XYZ[i][j-1][n4]
                else:
                    z = (XYZ[i][j][n4] + XYZ[i][j-1][n4])/2
                z_arr = np.append(z_arr,z)
                
        #check *horizontal pixel edges
        if (y_left_corner - vertex_array [k][l][n6])%pixel_height == 0:
            if (vertex_array [k][l][n5] - x_left_corner)%pixel_width != 0:
                
                if (vertex_array [k][l][n6] - y_left_corner) == 0:
                    z = XYZ[i][j][n4]	
                elif (y_left_corner - vertex_array [k][l][n6]) == number_ypix*pixel_height*(-1):
                    z = XYZ[i-1][j][n4]
                else:
                    z = (XYZ[i][j][n4] + XYZ[i-1][j][n4])/2
                z_arr = np.append(z_arr,z)
        
        #check *intersection of vertical and horizontal pixel edges
        if (vertex_array [k][l][n5] - x_left_corner)%pixel_width == 0:
            if (y_left_corner - vertex_array [k][l][n6])%pixel_height == 0:
                
                if (vertex_array [k][l][n5] - x_left_corner) == number_xpix*pixel_width:
                    if (y_left_corner - vertex_array [k][l][n6]) == number_ypix*pixel_height*(-1):
                        z = XYZ[i-1][j-1][n4]
                    elif (vertex_array [k][l][n6] - y_left_corner) == 0:
                        z = XYZ[i][j-1][n4]
                    else:
                        z = (XYZ[i][j-1][n4] + XYZ[i-1][j-1][n4])/2
                
                elif (y_left_corner - vertex_array [k][l][n6]) == number_ypix*pixel_height*(-1):
                    if (vertex_array [k][l][n5] - x_left_corner) == 0:
                        z = XYZ[i-1][j][n4]
                    else:
                        z = (XYZ[i-1][j][n4] + XYZ[i-1][j-1][n4])/2
                        
                else:
                    z = pixel_elevation(self,XYZ[i][j][n4],XYZ[i-1][j][n4],XYZ[i][j-1][n4],XYZ[i-1][j-1][n4])
                z_arr = np.append(z_arr,z)	
                                                    
    return z_arr


#create an empty type array 
def create_type_array(self):
    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float)]
    #array initialization and deleting items
    empty_new_vertex_arr= np.array([7*(0)], dtype=d_type)
    empty_new_vertex_arr = np.delete(empty_new_vertex_arr,0,0)
    return(empty_new_vertex_arr)


#return a sorted array by checking XY position
#"X1","X2","Y1","Y2" - points coordinates, "new_vertex_arr" - unsorted array 
def sort_lines_elements(self,X1,X2,Y1,Y2,new_vertex_arr):
    #the linear layer will be cut relative to the pixel position and not the linear layer, therefore it will be sorted according to the XY value of the original and new vertices
    if (X2>X1):
        std_new_vert_arr = np.sort(new_vertex_arr, order='x')
    elif(X2<X1):
        std_new_vert_arr = np.sort(new_vertex_arr, order='x')[::-1]   
    else:
        if (Y2>Y1):
            std_new_vert_arr = np.sort(new_vertex_arr, order='y')
        elif(Y2<Y1):
            std_new_vert_arr = np.sort(new_vertex_arr, order='y')[::-1]
    
    return std_new_vert_arr


#return an array with added elements with the exact type
#"k","l","n","x","y" - values, "new_vertex_arr" - array to be changed, "d_type" - array type
def add_array_element(self,k,l,n,m,x,y,new_vertex_arr,d_type):
    value_arr = np.array([(k,l,l+1,n,m,x,y)], dtype=d_type)
    new_vertex_arr = np.concatenate((new_vertex_arr,value_arr), axis=0)
    return new_vertex_arr


#find intersections between lines and pixels
#"a","b" - linear function parameters
def raster_line_intersection(self,lines,number_ypix,number_xpix,x_left_corner,y_left_corner,pixel_width,pixel_height,vertex_array):
    
    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float)]
    end_std_new_vert_arr = create_type_array(self)
    k=0
    
    for ft in lines.getFeatures():
        geom_ft = ft.geometry() 
        elem_xy = vector_objects_elements(self,ft)
        l=0

        for v in elem_xy:
            num_vert = int(len(v))
            
        for l in range(0, num_vert-1):	
            n=1
            
            #passing over each object, after each vertex up to the penultimate, the initial and final XY lines between the two existing vertices are determined
            X1 = vertex_array [k][l][2]
            Y1 = vertex_array [k][l][3]
            X2 = vertex_array [k][l+1][2]
            Y2 = vertex_array [k][l+1][3]
            x_max = max(X1,X2)
            x_min = min(X1,X2)
            y_max = max(Y1,Y2)
            y_min = min(Y1,Y2)
    
            new_vertex_arr = create_type_array(self)
            new_vertex_arr = add_array_element(self,k,l,0,1,X1,Y1,new_vertex_arr,d_type)
            
            #all vertical pixel edges are examined and checked using the formula for the linear function whether and where the line intersects the edge
            for j in range(0, number_xpix+1):
                x = x_left_corner + j*pixel_width
                 
                if (x>x_min) and (x<x_max):
                    if 0==(Y2-Y1):
                        y = Y1
                    else:
                        a = (Y2-Y1)/(X2-X1)
                        b = Y1-a*X1
                        if a==0:
                            y = Y1
                        else:
                            y = a*x+b
        
                    new_vertex_arr = add_array_element(self,k,l,n,0,x,y,new_vertex_arr,d_type)
                    n+=1
            
            #all horizontal pixel edges are examined and checked using the formula for the linear function whether and where the line intersects the edge		
            for i in range(0, number_ypix+1):
                y = y_left_corner + i*pixel_height
                            
                if (y>y_min) and (y<y_max):
                    if 0==(X2-X1):
                        x = X1
                    else:
                        a = (Y2-Y1)/(X2-X1)
                        b = Y1-a*X1	
                        
                        x =(y-b)/a

                    new_vertex_arr = add_array_element(self,k,l,n,0,x,y,new_vertex_arr,d_type)
                    n+=1
            
            std_new_vert_arr = sort_lines_elements(self,X1,X2,Y1,Y2,new_vertex_arr)
            end_std_new_vert_arr = np.concatenate((end_std_new_vert_arr,std_new_vert_arr), axis=0)
                
        end_std_new_vert_arr = add_array_element(self,k,l+1,n,1,X2,Y2,end_std_new_vert_arr,d_type)
        k+=1
    
    return end_std_new_vert_arr
    

#create an array with heights for every vertex/point from a linear layer
def new_vertex_elevation_array(self,end_std_new_vert_arr,XYZ,raster_param_arr):
    
    #create an array with information about existing and new vertices
    new_arr = [end_std_new_vert_arr]
    new_z_arr = np.array([[]],dtype='f')
    
    k=0
    elem_num = np.size(new_arr)
    for l in range (0,elem_num):
        new_z_arr = add_elevation(self,XYZ,new_arr,new_z_arr,k,l,4,5,6,raster_param_arr[4],raster_param_arr[3],raster_param_arr[5],raster_param_arr[6],raster_param_arr[7],raster_param_arr[8])
    
    return new_z_arr
    

#create a type array with infrmation about all vertices (first and last vertex for every line is original, others are new)
#"xy_round","z_round" - decimal places for x,y and z values
def new_vertex_information_array(self,end_std_new_vert_arr,new_z_arr,xy_round,z_round):
    
    #a new empty array with added "z" value
    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float), ('z', float)]
    new_info_arr = np.array([],dtype= d_type)
    
    elem_num = np.size(end_std_new_vert_arr)
    
    #create a sorted array
    for i in range (0,elem_num):
        value_arr = np.array([(end_std_new_vert_arr[i][0],end_std_new_vert_arr[i][1],end_std_new_vert_arr[i][2],i,end_std_new_vert_arr[i][4],round(end_std_new_vert_arr[i][5],xy_round),round(end_std_new_vert_arr[i][6],xy_round),round(new_z_arr[i],z_round))], dtype=d_type)
        new_info_arr = np.concatenate((new_info_arr,value_arr), axis=0)
            
    return new_info_arr


#return object as points
def points_objects_elements(self,ft):
    geom_ft = ft.geometry()
    geom_type = QgsWkbTypes.isSingleType(geom_ft.wkbType())
    
    #check the type of geometry: point, multipoint, else
    if geom_ft.type() == QgsWkbTypes.PointGeometry:
        if geom_type:
            elem_xy = geom_ft.asPoint()
        else:
            elem_xy = geom_ft.asMultiPoint()
    else:
        QMessageBox(QMessageBox.Warning, "Error","Invalid points layer format.", QMessageBox.Ok).exec()
    
    return elem_xy


#return an array with [points_id, x, y] 
def points_information_array(self,route_pts):
    
    d_type = [('obj_number', int), ('x', float), ('y', float)]
    route_pts_arr = np.array([],dtype= d_type)
    
    for ft in route_pts.getFeatures(): 
        object_number = ft.id()
        elem_xy = points_objects_elements(self,ft)

        val_x = elem_xy.x()
        val_y = elem_xy.y()
        
        value_arr = np.array([(object_number, val_x, val_y)], dtype=d_type)
        route_pts_arr = np.concatenate((route_pts_arr,value_arr), axis=0)
            
    return route_pts_arr
    

#return an array with the closest points
def closest_point(self,route_pts_arr,new_info_arr):
    
    d_type = [('pts_number', int), ('vrt_number', int), ('distance', float)]
    closest_pts_arr = np.array([],dtype= d_type)
    
    j=0
    #"pts" - points from the points layer, "vrt" - vertices from the linear layer, "dist" - distance between the specified point and vertex
    for pts in route_pts_arr:
        for vrt in new_info_arr:
            dist = math.sqrt((pts[1] - vrt[5])**2 + (pts[2] - vrt[6])**2)
            
            if vrt[3] == 0:
                #"closest_dist" closest distance between point and vertex
                closest_dist = dist
                value = np.array([(pts[0],vrt[3],dist)],dtype= d_type)
                closest_pts_arr = np.concatenate((closest_pts_arr,value), axis =0)
            
            elif dist < closest_dist: 
                i=0
                for el in closest_pts_arr:
                    if el [0] == j:
                        closest_pts_arr = np.delete(closest_pts_arr,i)
                        i-=1
                    i+=1
                
                closest_dist = dist
                value = np.array([(pts[0],vrt[3], dist)],dtype= d_type)
                closest_pts_arr = np.concatenate((closest_pts_arr,value), axis =0)
                
            elif dist == closest_dist:
                closest_dist = dist
                #"simil" - if there is the same distance between point and two different vertices
                simil = closest_pts_arr[closest_pts_arr['pts_number'] == pts[0]-1]
                simil = simil[simil['vrt_number'] == vrt[3]]
                
                if np.size(simil) == 0:
                    value = np.array([(pts[0],vrt[3], dist)],dtype= d_type)
                    closest_pts_arr = np.concatenate((closest_pts_arr,value), axis =0) 
        j+=1
            
    return closest_pts_arr


#return an array with the possible combinations of routes between the closest vertices
def closest_pts_possibility(self,closest_pts_arr,new_info_arr):
    
    #last and first point
    num_max = np.max(closest_pts_arr['pts_number'])
    num_min = np.min(closest_pts_arr['pts_number'])
    
    num=0
    #if there are pairs of vertices with the same distance from the same point and they have the same x,y - delete one of them
    for el in closest_pts_arr:
        num+=1
        if np.any((closest_pts_arr[:] == el), axis=0):
            arr = closest_pts_arr[closest_pts_arr['pts_number'] == el[0]]
            arr = arr[arr['vrt_number'] != el[1]]
            x = new_info_arr[el[1]][5]
            y = new_info_arr[el[1]][6]
            
            for ele in arr:
                x_e = new_info_arr[ele[1]][5]
                y_e = new_info_arr[ele[1]][6]
                
                if (x == x_e) and (y == y_e):
                    closest_pts_arr = np.delete(closest_pts_arr,[num])
                    num-=1
        else:
            pass
            
    clos_poss_arr = []
    #find all combinations of routes between the closest vertices, "it" - all combinations to check and optimize
    it = itertools.combinations(closest_pts_arr,num_max+1)
    
    for elem in it:
        if elem[0][0] == num_min:
            cond = True
            
            for i in range(0,num_max-1):			
                #check if next element is a next point, True -> append element, False -> pass
                if elem[i][0] == elem[i+1][0]-1:
                    pass
                else:
                    cond = False
            if cond:
                clos_poss_arr.append(elem)
    
    return clos_poss_arr


#removes a specified element from the table relative to the given parameters
#"param1" - characteristics of the analysis place, "param2" - the given parameter, if "elem[param1]" == param2, then the element is removed
def delete_arr_elem(self,array_in,param1,param2):
    i=0
    for elem in array_in:
        if elem[param1]== param2:
            array_in = np.delete(array_in ,i)
            i-=1
        i+=1
    return array_in


#calculate the next original vertex relative to the element number: "ver_num" from the "new_info_arr"
def next_origin_vertex(self,ver_num,new_info_arr,ob_num):
    next_vrt = new_info_arr[new_info_arr['obj_number'] == ob_num]
    next_vrt = next_vrt[next_vrt['vert_number'] == ver_num+1]
    next_vrt = next_vrt[next_vrt['origin'] == 1]
    return next_vrt


#calculate the previous vertex
def prev_origin_vertex(self,ver_num,new_info_arr,ob_num,origin_num):
        
    #present vertex
    pres_vrt = new_info_arr[new_info_arr['obj_number'] == ob_num]
    pres_vrt = pres_vrt[pres_vrt['vert_number'] == ver_num]
    pres_vrt = pres_vrt[pres_vrt['origin'] == 1]
            
    #previous vertex
    prev_vrt = new_info_arr[new_info_arr['obj_number'] == ob_num]
    prev_vrt = prev_vrt[prev_vrt['vert_number'] == ver_num-1]
    prev_vrt = prev_vrt[prev_vrt['origin'] == 1]

    #check if the element is one of the origin vertex, if true -> do nothing, if not -> previous vertex is  the first present vertex
    if origin_num != 1:
        prev_vrt = pres_vrt
            
    return prev_vrt


#checks if the next/previous element exists, if exists -> returns: True and the value of the next element, if not -> it returns: False
def check_elem_exist(self,array_in,param1,param2,param_np,ver_num,new_info_arr,ob_num,origin_num):
    try:
        if param_np == "next":
            next_prev_vrt = next_origin_vertex(self,ver_num,new_info_arr,ob_num)
            
        if param_np == "prev":
            next_prev_vrt = prev_origin_vertex(self,ver_num,new_info_arr,ob_num,origin_num)

        cond = True
    except:
        next_prev_vrt = 0
        cond = False
        array_in = delete_arr_elem(self,array_in, param1,param2)

    if np.size(next_prev_vrt) == 0:
        cond = False
        array_in = delete_arr_elem(self,array_in, param1,param2)
    
    return next_prev_vrt, cond, array_in


#check if the route passes through the same vertex again (or vertex with the same x,y value), "dup" - an array with duplicate values
def duplicate_vertex(self,array_in,next_prev_vrt,i):
    dup = array_in[array_in['x'] == next_prev_vrt[0]['x']]
    dup = dup[dup['y'] == next_prev_vrt[0]['y']]
    dup = dup[dup['id'] == i]
    return dup


#add an element to a type array
def add_next_elem_array(self,array_in,next_vrt,d_type,i,ver_num_end,ver_num2_end,route_num):
    value = np.array([(next_vrt[0][0],next_vrt[0][1], next_vrt[0][2],next_vrt[0][3],next_vrt[0][4],next_vrt[0][5],next_vrt[0][6],next_vrt[0][7],i,ver_num_end,ver_num2_end,route_num)],dtype= d_type)
    array_in = np.concatenate((array_in,value), axis =0)
    return array_in


#find elements with the "id", copy them and change value to "id"+"dif" (difference)
def add_equal_id(self,array_in,id,dif):    
    value = array_in[array_in['id'] == id]
    value ['id'] = id+dif
    array_in = np.concatenate((array_in,value), axis =0)
    return array_in


#find original vertices with the same value x,y, then return an array with these values 
def other_objects(self,new_info_arr,next_vrt,ver_num_end):

    #choose elements with the same x,y values for "next_vrt" - currently checked item
    new_ob = new_info_arr[new_info_arr['x'] == next_vrt[0]['x']]
    new_ob = new_ob[new_ob['y'] == next_vrt[0]['y']]
    
    #choose element with the "ver_num_end", then check if there are more original vertices with the same x,y, 
    #check first vertex from the currently checked pair
    n1 = new_info_arr[new_info_arr['new_vert_number'] == ver_num_end]
    n2 = new_info_arr[new_info_arr['x'] == n1[0]['x']]
    n2 = n2[n2['y'] == n1[0]['y']]
    n2 = n2[n2['new_vert_number'] != n1[0]['new_vert_number']]
    
    #choose if original and different from checked item
    new_ob = new_ob[new_ob['origin'] == 1]
    new_ob = new_ob[new_ob['new_vert_number'] != next_vrt[0]['new_vert_number']]
    new_ob = new_ob[new_ob['new_vert_number'] != ver_num_end]
    #choose if different from first vertex from the currently checked pair
    if np.size(n2) !=0:
        new_ob = new_ob[new_ob['new_vert_number'] != n2[0]['new_vert_number']]

    return new_ob


#if current vertex is last from the original (before the last from currently checked pair), then calculate all intermediate vertices
def calc_last_vert(self,arr_next,new_info_arr,ver_num_next,ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end2):
    while ver_num_next != ver_num2_end:
        ver_num_next+=1
        value = np.array([(new_info_arr[ver_num_next][0], new_info_arr[ver_num_next][1], new_info_arr[ver_num_next][2], new_info_arr[ver_num_next][3], new_info_arr[ver_num_next][4], new_info_arr[ver_num_next][5],new_info_arr[ver_num_next][6],new_info_arr[ver_num_next][7],i,ver_num_end,ver_num2_end2,route_num)],dtype= d_type)
        arr_next = np.concatenate((arr_next,value), axis =0)
    return arr_next


#if current vertex is last from the original (before the last from currently checked pair), then calculate all intermediate vertices  
def calc_last_vert_prev(self,arr_next,new_info_arr,ver_num_next,ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end2):
    while ver_num_next != ver_num2_end:
        ver_num_next-=1
        value = np.array([(new_info_arr[ver_num_next][0], new_info_arr[ver_num_next][1], new_info_arr[ver_num_next][2], new_info_arr[ver_num_next][3], new_info_arr[ver_num_next][4], new_info_arr[ver_num_next][5],new_info_arr[ver_num_next][6],new_info_arr[ver_num_next][7],i,ver_num_end,ver_num2_end2,route_num)],dtype= d_type)
        arr_next = np.concatenate((arr_next,value), axis =0)
    return arr_next


#add an element to a type array
def add_type_element_arr(self,array_in,el,d_type,i,ver_num_end,ver_num2_end,route_num):
    value = np.array([(el[0],el[1],el[2],el[3],el[4],el[5],el[6],el[7],i,ver_num_end,ver_num2_end,route_num)],dtype = d_type)
    array_in = np.concatenate((array_in,value), axis =0)	
    return array_in
    

#check next vertex and possible routes for the currently checked pair of vertices
def next_vert2(self,i,array_next,ver_num,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num):
    
    #if next vertex is equal to the second vertex -> pass and stop checking
    if next_vrt_calc == ver_num2_end:
        next_vrt_calc = -1
        pass
        
    else:
        ver_num_next = ver_num
        ver_num_next += 1

        col = check_elem_exist(self,array_next,"id",i,"next",ver_num,new_info_arr,ob_num,origin_num)
        next_vrt = col[0]
        cond = col[1]
        array_next = col[2]

        #if next element exists
        if cond==True:
            dup = duplicate_vertex(self,array_next,next_vrt,i)
            
            #if route has the same point more than once -> delete this route
            if np.size(dup) !=0:
                array_next = delete_arr_elem(self,array_next,"id",i)

            else:
                array_next = add_next_elem_array(self,array_next,next_vrt,d_type,i,ver_num_end,ver_num2_end,route_num)
                j=i
                
                #check if next vertex is equals to the second vertex
                if next_vrt[0][3] == ver_num2_end:
                    array_next = add_next_elem_array(self,array_next,next_vrt,d_type,i,ver_num_end,ver_num2_end,route_num)
                    pass

                else:
                    #check if current vertex has equivalent with the same x,y, but from the other object 
                    new_ob = other_objects(self,new_info_arr,next_vrt,ver_num_end)
                    
                    #if there is no equivalent -> check if vertex belongs to the same line
                    if np.size(new_ob) == 0:
                        if next_vrt[0][0] == ob_num2:
                            if next_vrt[0][1] == ver_num2:
                                array_next = calc_last_vert(self,array_next,new_info_arr,next_vrt[0][3],ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end)
                                #assign value in order to stop process
                                next_vrt_calc = ver_num2_end
                                
                    #if there is an equivalent -> treat as next vertex to explore
                    else: 
                        for el in new_ob:
                            ob_num4 = el[0]
                            ver_num4 = el[1]
                            origin_num4 = el[4]
                            max_id = np.max(array_next['id'])
                            array_next = add_equal_id(self,array_next,i,max_id-i+1)
                            i = max_id+1				
                            
                            #check if vertex belongs to the same line and then calculate intermediate points
                            if (ob_num4 == ob_num2) and (ver_num4 == ver_num2):
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)
                                array_next = calc_last_vert(self,array_next,new_info_arr,el[3],ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end)

                            else:
                                #if not belongs to the same line, check the next vertex
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)
                                array_next = next_vert2(self,i,array_next,ver_num4,ver_num2_end,ob_num4,origin_num4,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
                                
                                #if not belongs to the same line, check also the previous vertex
                                max_id = np.max(array_next['id'])
                                i = max_id+1
                                array_next = add_equal_id(self,array_next,j,max_id-j+1)
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)
                                array_next = prev_vert2(self,i,array_next,ver_num4,ver_num2_end,ob_num4,origin_num4,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
                        
                        #check again if next vertex belongs to the same line
                        if next_vrt[0][0] == ob_num2:
                            if next_vrt[0][1] == ver_num2:
                                array_next = calc_last_vert(self,array_next,new_info_arr,next_vrt[0][3],ver_num2_end,j,d_type,ver_num_end,route_num,ver_num2_end)
                                next_vrt_calc = ver_num2_end
                    
                    #call the function again in order to check next vertices
                    array_next = next_vert2(self,j,array_next,ver_num_next,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
                    
    return array_next


#check previous vertex ("next_vrt" the name still the same like in the "next_vert2") and possible routes for the currently checked pair of vertices
def prev_vert2(self,i,array_next,ver_num,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num):
    
    #if previous vertex is equal to the second vertex -> pass and stop checking
    if next_vrt_calc == ver_num2_end:
        next_vrt_calc = -1
        pass
        
    else:
        ver_num_next = ver_num
        ver_num_next-=1

        col = check_elem_exist(self,array_next,"id",i,"prev",ver_num,new_info_arr,ob_num,origin_num)
        next_vrt = col[0]
        cond = col[1]
        array_next = col[2]
        
        #if next previous element exists
        if cond==True:
            dup = duplicate_vertex(self,array_next,next_vrt,i)
            
            #if route has the same point more than once -> delete this route
            if np.size(dup) !=0:
                array_next = delete_arr_elem(self,array_next,"id",i)
                
            else:
                j=i
                
                #check if previous vertex is equals to the second vertex
                if next_vrt[0][3] == ver_num2_end:
                    array_next = add_next_elem_array(self,array_next,next_vrt,d_type,i,ver_num_end,ver_num2_end,route_num)
                    pass

                else:
                    array_next = add_next_elem_array(self,array_next,next_vrt,d_type,i,ver_num_end,ver_num2_end,route_num)
                    new_ob = other_objects(self,new_info_arr,next_vrt,ver_num_end)
                    
                    #if there is no equivalent -> check if vertex belongs to the same line or the next line of the same object                        
                    if np.size(new_ob) == 0:
                        if next_vrt[0][0] == ob_num2:
                            #check if it is the next line
                            if next_vrt[0][1] == ver_num2+1:
                                if ver_num != ver_num2:
                                    arr_size = np.size(array_next)
                                    array_next = calc_last_vert_prev(self,array_next,new_info_arr,array_next[arr_size-1][3],ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end)
                                    #assign value in order to stop process
                                    next_vrt_calc = ver_num2_end
                            
                            #check if it is the same line
                            elif next_vrt[0][1] == ver_num2:
                                if ver_num != ver_num2:
                                    array_next = array_next[:-1]
                                    arr_size = np.size(array_next)
                                    array_next = calc_last_vert_prev(self,array_next,new_info_arr,array_next[arr_size-1][3],ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end)
                                    #assign value in order to stop process
                                    next_vrt_calc = ver_num2_end
                    
                    #if there is an equivalent -> treat as the next vertex to explore
                    else: 
                        for el in new_ob:
                            ob_num4 = el[0]
                            ver_num4 = el[1]
                            origin_num4 = el[4]
                            max_id = np.max(array_next['id'])
                            array_next = add_equal_id(self,array_next,i,max_id-i+1)
                            i = max_id+1
                            
                            #check if vertex belongs to the line and then calculate intermediate points
                            if (ob_num4 == ob_num2) and (ver_num4 == ver_num2):
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)
                                array_next = calc_last_vert(self,array_next,new_info_arr,el[3],ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end)

                            else:
                                #if not belongs to the same line, check the previous vertex
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)
                                array_next = prev_vert2(self,i,array_next,ver_num4,ver_num2_end,ob_num4,origin_num4,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
                                
                                #if not belongs to the same line, check also the next vertex
                                max_id = np.max(array_next['id'])
                                i = max_id+1
                                array_next = add_equal_id(self,array_next,j,max_id-j+1)
                                array_next = add_type_element_arr(self,array_next,el,d_type,i,ver_num_end,ver_num2_end,route_num)	
                                array_next = next_vert2(self,i,array_next,ver_num4,ver_num2_end,ob_num4,origin_num4,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
                        
                        #check again if previous vertex belongs to the same line
                        if next_vrt[0][0] == ob_num2:
                            if next_vrt[0][1] == ver_num2+1:
                                array_next = calc_last_vert_prev(self,array_next,new_info_arr,next_vrt[0][3],ver_num2_end,j,d_type,ver_num_end,route_num,ver_num2_end)
                                next_vrt_calc = ver_num2_end	
        
                    #call the function again in order to check next previous vertices
                    array_next = prev_vert2(self,j,array_next,ver_num_next,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
    
    return array_next


#calculate intermediate points -> forward or backwards
def identical_line(self,array_next,new_info_arr,ver_num_next,ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end2):
        
    if ver_num_end < ver_num2_end:
        array_next = calc_last_vert(self,array_next,new_info_arr,ver_num_next,ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end2)
    
    elif ver_num_end > ver_num2_end:
        array_next = calc_last_vert_prev(self,array_next,new_info_arr,ver_num_next,ver_num2_end,i,d_type,ver_num_end,route_num,ver_num2_end2)
    else:
        print("The same vertex for two points")
        pass

    return array_next



#return an array with route options, used in "possible_routes"
def calc_routes(self,ob_num,ob_num2,ver_num,ver_num2,new_info_arr,ver_num_end,d_type,id,ver_num2_end,route_num,possible_routes_arr,origin_num,array_next):
    
    #if the first and second vertex lies on the same line -> calculate intermediate points
    if (ob_num == ob_num2) and (ver_num == ver_num2):
        array_next = np.array([],dtype= d_type)
        array_next = add_type_element_arr(self,array_next,new_info_arr[ver_num_end],d_type,id,ver_num_end,ver_num2_end,route_num)
                
        arr_pres = identical_line(self,array_next,new_info_arr,ver_num_end,ver_num2_end,id,d_type,ver_num_end,route_num,ver_num2_end)
        possible_routes_arr = np.concatenate((possible_routes_arr,arr_pres), axis =0)
    
    #for the next element        
    try:
        id = np.max(possible_routes_arr['id'])+1
    except:
        id=0
    
    next_vrt_calc = -1
    array_next = np.array([],dtype= d_type)
    #add first element for the current route
    array_next = add_type_element_arr(self,array_next,new_info_arr[ver_num_end],d_type,id,ver_num_end,ver_num2_end,route_num)
    
    #check the next element
    arr_n = next_vert2(self,id,array_next,ver_num,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
    possible_routes_arr = np.concatenate((possible_routes_arr,arr_n), axis =0)			
                
    #for the previous element
    try:
        id = np.max(possible_routes_arr['id'])+1
    except:
        id=0
        
    array_next = np.array([],dtype= d_type)
    #add first element for the current route
    array_next = add_type_element_arr(self,array_next,new_info_arr[ver_num_end],d_type,id,ver_num_end,ver_num2_end,route_num)
    
    #check the previous element
    arr_p = prev_vert2(self,id,array_next,ver_num,ver_num2_end,ob_num,origin_num,new_info_arr,d_type,ob_num2,ver_num2,next_vrt_calc,ver_num_end,route_num)
    possible_routes_arr = np.concatenate((possible_routes_arr,arr_p), axis =0)

    return possible_routes_arr


#return an array with the final route options
def possible_routes(self,closest_pts_arr,new_info_arr):
    
    clos_arr_size = len(closest_pts_arr)
    route_num =0

    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float), ('z', float), ('id', int),('num_from', int),('num_to', int),('route_num', int)]
    array_next = np.array([],dtype= d_type)
    possible_routes_arr = np.array([],dtype= d_type)
    
    #for all route numbers, for all closest vertices from the route
    for j in range (0,clos_arr_size):
        clos_arr_size2 = len(closest_pts_arr[j])
        for i in range (0,clos_arr_size2-1):
            
            #first vertex to explore
            elem = closest_pts_arr[j][i]
            ob_num = new_info_arr[elem[1]] [0]
            ver_num = new_info_arr[elem[1]] [1]
            ver_num_end = new_info_arr[elem[1]] [3]
            origin_num = new_info_arr[elem[1]] [4]
            
            #second vertex to explore
            elem2 = closest_pts_arr[j][i+1]
            ob_num2 = new_info_arr[elem2[1]] [0]
            ver_num2 = new_info_arr[elem2[1]] [1]
            ver_num2_end = new_info_arr[elem2[1]] [3]
            
            try:
                id = np.max(possible_routes_arr['id'])+1
            except:
                id=0
                    
            #calculate route options
            possible_routes_arr = calc_routes(self,ob_num,ob_num2,ver_num,ver_num2,new_info_arr,ver_num_end,d_type,id,ver_num2_end,route_num,possible_routes_arr,origin_num,array_next)
            
            #check if the first vertex has equivalent in other objects
            first_dupli = new_info_arr[new_info_arr['x'] == new_info_arr[ver_num_end]['x']]
            first_dupli = first_dupli[first_dupli['y'] == new_info_arr[ver_num_end]['y']]
            first_dupli = first_dupli[first_dupli['new_vert_number'] != new_info_arr[ver_num_end]['new_vert_number']]
            
            #if there is an equivalent -> calculate routes for them all 
            for elemen in first_dupli:
                #change vertex and explore
                ob_num = elemen [0]
                ver_num = elemen [1]
                ver_num_end = elemen [3]
                origin_num = elemen [4]

                try:
                    id = np.max(possible_routes_arr['id'])+1
                except:
                    id=0

                array_next = np.array([],dtype= d_type)
                array_next = add_type_element_arr(self,array_next,new_info_arr[ver_num_end],d_type,id,ver_num_end,ver_num2_end,route_num)
                
                possible_routes_arr = calc_routes(self,ob_num,ob_num2,ver_num,ver_num2,new_info_arr,ver_num_end,d_type,id,ver_num2_end,route_num,possible_routes_arr,origin_num,array_next)
                
        route_num+=1
        
    return possible_routes_arr


#return an array with complete route points content
def complete_possible_route(self,possible_routes_arr,new_info_arr):
    
    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float), ('z', float), ('id', int),('num_from', int),('num_to', int),('route_num', int)]
    compl_poss_route_arr = np.array([],dtype= d_type)
    try:
        num_min = np.min(possible_routes_arr['route_num'])
        num_max = np.max(possible_routes_arr['route_num'])
    except:
        QMessageBox(QMessageBox.Warning, "Error","Cannot find a solution.", QMessageBox.Ok).exec()
    
    #for all possible routes calculate intermediate points
    for i in range (num_min,num_max+1):
        main_route_arr = possible_routes_arr[possible_routes_arr['route_num'] == i]
        route_num = i

        size = np.size(main_route_arr)
        
        #for elements in the array check neighboring elements
        for i in range(0,size-1):

            first_ob = main_route_arr[i][0]
            next_ob = main_route_arr[i+1][0]
            
            first_vert = main_route_arr[i][3]
            next_vert = main_route_arr[i+1][3]
            
            first_id = main_route_arr[i][8]
            next_id = main_route_arr[i+1][8]
            
            first_from = main_route_arr[i][9]
            next_from = main_route_arr[i+1][9]
            
            first_to = main_route_arr[i][10]
            next_to = main_route_arr[i+1][10]

            #if: the object, fragmentary route and points pair is the same, but vertices are different
            if (first_vert != next_vert) and (first_ob == next_ob) and (first_id == next_id) and (first_from == next_from) and (first_to == next_to):
                siz = np.size(compl_poss_route_arr)
                
                if siz != 0:
                    #if current and previous elements are the same
                    if main_route_arr[i][3] == compl_poss_route_arr[siz-1][3]:
                        pass
                    else:
                        compl_poss_route_arr = add_type_element_arr(self,compl_poss_route_arr,main_route_arr[i],d_type,main_route_arr[i][8],main_route_arr[i][9],main_route_arr[i][10],route_num)
                else:
                    #if it is the first explored vertex
                    compl_poss_route_arr = add_type_element_arr(self,compl_poss_route_arr,main_route_arr[i],d_type,main_route_arr[i][8],main_route_arr[i][9],main_route_arr[i][10],route_num)
                
                #calculate intermediate points
                if first_vert < next_vert:
                    compl_poss_route_arr = calc_last_vert(self,compl_poss_route_arr,new_info_arr,first_vert,next_vert,main_route_arr[i][8],d_type,main_route_arr[i][9],route_num,main_route_arr[i][10])
                else:
                    compl_poss_route_arr = calc_last_vert_prev(self,compl_poss_route_arr,new_info_arr,first_vert,next_vert,main_route_arr[i][8],d_type,main_route_arr[i][9],route_num,main_route_arr[i][10])
                
            #if vertices are the same or one of them (object, fragmentary route, points pair) is not the same       
            else:
                siz2 = np.size(compl_poss_route_arr)
                #if current and previous elements are the same
                if main_route_arr[i][3] == compl_poss_route_arr[siz2-1][3]:
                    pass
                else:
                    compl_poss_route_arr = add_type_element_arr(self,compl_poss_route_arr,main_route_arr[i],d_type,main_route_arr[i][8],main_route_arr[i][9],main_route_arr[i][10],route_num)
            
    return compl_poss_route_arr


def check_array_shortages(self,compl_poss_route_arr,possible_routes_arr):
    #check if there is no elements shortages in first elements of every new fragmentary route
    d_type = [('obj_number', int), ('vert_number', int), ('next_vert_number', int), ('new_vert_number', int), ('origin', int), ('x', float), ('y', float), ('z', float), ('id', int),('num_from', int),('num_to', int),('route_num', int)]
    new_compl_poss_route_arr = np.array([],dtype= d_type)
    siz = np.size(compl_poss_route_arr)
    for j in range (0,siz-1):
        eleme = compl_poss_route_arr[j]
        eleme2 = compl_poss_route_arr[j+1]
        new_compl_poss_route_arr = add_type_element_arr(self,new_compl_poss_route_arr,compl_poss_route_arr[j],d_type,compl_poss_route_arr[j][8],compl_poss_route_arr[j][9],compl_poss_route_arr[j][10],compl_poss_route_arr[j][11])
        
        #check if there is a shortage of the first element and add missing element if it is needed
        if (eleme[8] != eleme2[8]) and (eleme2[3] != eleme2[9]):
            element = possible_routes_arr[possible_routes_arr['new_vert_number'] == eleme2[9]]
            new_compl_poss_route_arr = add_type_element_arr(self,new_compl_poss_route_arr,element[0],d_type,eleme2[8],eleme2[9],eleme2[10],eleme2[11])
        #add last element
        if j == siz-2:
            new_compl_poss_route_arr = add_type_element_arr(self,new_compl_poss_route_arr,compl_poss_route_arr[j+1],d_type,compl_poss_route_arr[j+1][8],compl_poss_route_arr[j+1][9],compl_poss_route_arr[j+1][10],compl_poss_route_arr[j+1][11])
    
    return new_compl_poss_route_arr


#add an element to a type array    
def add_type_element_section(self,array_in,d_type,route_num,el_0,el_1,el_2,el_3,el_5,el_6,el_7,el_8,el_9,el_10,el_11,el_12):
    value = np.array([(el_0,el_1,el_2,el_3,route_num,el_5,el_6,el_7,el_8,el_9,el_10,el_11,el_12)],dtype = d_type)
    array_in = np.concatenate((array_in,value), axis =0)
    return  array_in


#calculate time
def calculate_time(self,dist_3D,slope,elev):
    cond1 = self.dlg.veliocityLE.text()
    cond2 = self.dlg.methodMoveCB.currentText()
    if cond1 != "":
        try:
            #try to change string into float and calculate time
            cond1 = float(cond1)
            vel = cond1*1000/60
            time = dist_3D/vel
        except:
            QMessageBox(QMessageBox.Warning, "Error","Invalid veliocity value.", QMessageBox.Ok).exec()
            
    else:
        if cond2 == "foot-walking":
            #m/min, time in minutes
            vel = 5*1000/60
            #if slope is bigger than 12 degree, velioity is slower
            if abs(slope) > 12:
                time = dist_3D/vel + abs(elev/10)
            else:
                time = dist_3D/vel
        elif cond2 == "cycling":
            vel = 15*1000/60
            if slope > 12:
                time = dist_3D/vel + abs(elev/10)
            elif slope < -12:
                time = dist_3D/vel - abs(elev/10)
            else:
                time = dist_3D/vel
        elif cond2 == "car-driving":
            vel = 40*1000/60
            time = dist_3D/vel

    return time


#calculate mean slope and vertex position
def calculate_slope_vrt_position(self,slope_arr,elem_arr,elem):
    
    mean_slope = np.mean(slope_arr)
    pos_from = np.min(elem_arr)
    pos_to = np.max(elem_arr)+1
    vrt_first = elem[9]
    vrt_last = elem[10]
    
    return mean_slope, pos_from, pos_to, vrt_first, vrt_last

    
#calculate attributes for all routes between pairs of points
def route_sections_attributes(self,new_compl_poss_route_arr):

    d_type = [('pos_from', int),('pos_to', int),('vrt_first', int),('vrt_last', int),('route_num', int),('elevation', float),('length_2D', float),('length_3D', float),('approaches', float),('height_differences', float),('descents', float),('mean_slope', float),('time', float)]
    sections_attr_arr = np.array([],dtype= d_type)
    
    size = np.size(new_compl_poss_route_arr)
    elevation, elevation_plus, elevation_minus, length_2D, length_3D, approaches, descents, height_differences, time, elem_pos, route_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    slope_arr = np.array([])
    elem_arr = np.array([])
    
    #for every points pairs in "new_compl_poss_route_arr" calculate attributes
    for i in range (0,size-1): 
        
        #choose a pair of points
        elem = new_compl_poss_route_arr[i]
        elem2 = new_compl_poss_route_arr[i+1]
        #if a pair of next points with the same: route numbers and pair of points
        if (elem[8] == elem2[8]) and (elem[9] == elem2[9]) and (elem[10] == elem2[10]) and (elem[11] == elem2[11]):
            #pass calculations for the points with the same horizontal (x,y) position
            if (elem[5] == elem2[5]) and (elem[6] == elem2[6]):
                pass	
            else:
                elem_pos = i
                elem_arr = np.append(elem_arr, elem_pos)
                                        
                #summarize attributes: "elevation", "length_2D", "length_3D"
                elem_elevation = elem2[7] - elem[7]
                elevation = elevation + elem_elevation
                    
                elem_length_2D = math.sqrt((elem2[5] - elem[5])**2+(elem2[6] - elem[6])**2)
                length_2D = length_2D + elem_length_2D
                    
                elem_length_3D = math.sqrt((elem_length_2D )**2+(elem_elevation)**2)
                length_3D = length_3D + elem_length_3D
                
                #calculate approaches, descents, height_differences
                if elem_elevation > 0:
                    #"appraches" - calculate a distance with positive elevation
                    approaches = approaches + elem_length_3D
                    elevation_plus = elevation_plus + elem_elevation
                    #"height_differences" - added all elevation differences 
                    height_differences = height_differences + elem_elevation
                elif elem_elevation < 0:
                    #"descents" - calculate a distance with negative elevation
                    descents = descents + elem_length_3D
                    elevation_minus = elevation_minus + elem_elevation
                    #"height_differences" - added all elevation differences 
                    height_differences = height_differences + abs(elem_elevation)
  
                #slope in degree, absolute value for the mean slope  
                elem_slope = math.atan(elem_elevation/elem_length_2D)*180/math.pi
                slope_arr = np.append(slope_arr,abs(elem_slope))
                    
                # calculate "time"
                elem_time = calculate_time(self,elem_length_3D,elem_slope,elem_elevation)
                time = time + elem_time

                #save all added attributes to array when last element
                if i == size-2:
                    slope_vrt_pos = calculate_slope_vrt_position(self,slope_arr,elem_arr,elem)
                    sections_attr_arr = add_type_element_section(self,sections_attr_arr,d_type,route_num,slope_vrt_pos[1],slope_vrt_pos[2],slope_vrt_pos[3],slope_vrt_pos[4],elevation,length_2D,length_3D,approaches,height_differences,descents,slope_vrt_pos[0],time)
                        
                    elevation, elevation_plus, elevation_minus, length_2D, length_3D, approaches, descents, time  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    slope_arr = np.array([])
                    elem_arr = np.array([])
                    
                
        #if the same main route
        elif elem[11] == elem2[11]:
            #save all added attributes to array
            slope_vrt_pos = calculate_slope_vrt_position(self,slope_arr,elem_arr,elem)
            sections_attr_arr = add_type_element_section(self,sections_attr_arr,d_type,route_num,slope_vrt_pos[1],slope_vrt_pos[2],slope_vrt_pos[3],slope_vrt_pos[4],elevation,length_2D,length_3D,approaches,height_differences,descents,slope_vrt_pos[0],time)
                
            elevation, elevation_plus, elevation_minus, length_2D, length_3D, approaches, descents, time  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            slope_arr = np.array([])
            elem_arr = np.array([])
        
        else:
            #if next route
            slope_vrt_pos = calculate_slope_vrt_position(self,slope_arr,elem_arr,elem)		
            sections_attr_arr = add_type_element_section(self,sections_attr_arr,d_type,route_num,slope_vrt_pos[1],slope_vrt_pos[2],slope_vrt_pos[3],slope_vrt_pos[4],elevation,length_2D,length_3D,approaches,height_differences,descents,slope_vrt_pos[0],time)		
            
            elevation, elevation_plus, elevation_minus, length_2D, length_3D, approaches, descents, time  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            slope_arr = np.array([])
            elem_arr = np.array([])
            route_num+=1
    
    return sections_attr_arr


#return an optimized array
def route_optimization(self,sections_attr_arr,closest_pts_arr0):

    #choose parameter to optimize route
    optimization = self.check_optimization()
    if optimization == "none":
        QMessageBox(QMessageBox.Warning, "Error","Optimization method not selected.", QMessageBox.Ok).exec()
    
    d_type = [('pos_from', int),('pos_to', int),('vrt_first', int),('vrt_last', int),('route_num', int),('elevation', float),('length_2D', float),('length_3D', float),('approaches', float),('height_differences', float),('descents', float),('mean_slope', float),('time', float)]
    route_optim_arr = np.array([],dtype= d_type)
    d_type2 = [('route_num', int),('elevation', float),('length_2D', float),('length_3D', float),('approaches', float),('height_differences', float),('descents', float),('mean_slope', float),('time', float)]
    route_optim_arr3 = np.array([],dtype= d_type2)
    d_type3 = [('route_num', int),('pos_from', int),('pos_to', int)]
    from_to_arr = np.array([],dtype= d_type3)
    
    #change "sections_attr_arr" by adding info about point number for vertex from "closest_pts_arr0"
    i=0
    for elem in sections_attr_arr:
        vrt_num = closest_pts_arr0[closest_pts_arr0['vrt_number'] == elem[2]]
        vrt_num = vrt_num[0][0] 
        sections_attr_arr[i][2] = vrt_num
        vrt_num2 = closest_pts_arr0[closest_pts_arr0['vrt_number'] == elem[3]]
        vrt_num2 = vrt_num2[0][0]
        sections_attr_arr[i][3] = vrt_num2
        i+=1
    
    for elem in sections_attr_arr:
        route_optim = sections_attr_arr[sections_attr_arr['vrt_first'] == elem[2]]
        route_optim = route_optim [route_optim ['vrt_last'] == elem[3]]
        route_optim = route_optim [route_optim ['route_num'] == elem[4]]
        
        #Choose maxmization or minimization
        cond = self.check_min_max()
        if cond == "min":
            route_min = np.min(route_optim[optimization])
            route_optim = route_optim [route_optim [optimization] == route_min]
        elif cond == "max":
            route_max = np.max(route_optim[optimization])
            route_optim = route_optim [route_optim [optimization] == route_max]
        
        route_optim_arr = np.concatenate((route_optim_arr,route_optim), axis =0)
    route_optim_arr = np.unique(route_optim_arr)
    
    route_num = np.max(route_optim_arr['route_num'])
    for i in range(0,route_num+1):

        route_optim_arr2 = route_optim_arr[route_optim_arr['route_num'] == i]
        
        #calculate attributes for the choosed route
        for elemen in route_optim_arr2:
            value = np.array([(elemen[4],elemen[0],elemen[1])],dtype = d_type3)
            from_to_arr = np.concatenate((from_to_arr,value),axis =0)
        #change a structured array -> to an unstructured array
        route_optim_arr2 = np.lib.recfunctions.structured_to_unstructured(route_optim_arr2)
        #the mean value of attributes
        route_optim_arr_mean = np.mean(route_optim_arr2, axis=0)
        #the sum of attributes
        route_optim_arr2 = np.sum(route_optim_arr2, axis=0)
        
        #append all attributes
        value1 = np.array([(i,route_optim_arr2[5],route_optim_arr2[6],route_optim_arr2[7],route_optim_arr2[8],route_optim_arr2[9],route_optim_arr2[10],route_optim_arr_mean[11],route_optim_arr2[12])],dtype = d_type2)
        route_optim_arr3 = np.concatenate((route_optim_arr3,value1), axis =0)
    
    return route_optim_arr3, from_to_arr


#return an array with route points
def points_set(self,new_compl_poss_route_arr,pos_from,pos_to,pts_arr,z_arr):	
    
    #sort points
    if pos_from > pos_to:
        for i in range (pos_from,pos_to-1,-1):
            x = new_compl_poss_route_arr[i][5]
            y = new_compl_poss_route_arr[i][6]
            z = new_compl_poss_route_arr[i][7]
            pt = QgsPointXY(x,y)
            z_arr = np.append(z_arr,z)
            pts_arr.append(pt)
            
    if pos_from <= pos_to:
        for i in range (pos_from,pos_to+1):
            x = new_compl_poss_route_arr[i][5]
            y = new_compl_poss_route_arr[i][6]
            z = new_compl_poss_route_arr[i][7]
            pt = QgsPointXY(x,y)
            z_arr = np.append(z_arr,z)
            pts_arr.append(pt)
    
    return z_arr, pts_arr


#create and display an output file
def create_shp_route(self,length_2D,length_3D,time,elevation,height_differences,approaches,descents,mean_slope,result_file_path,EPSG_num,route_optim_array,new_compl_poss_route_arr,from_to_arr,raster_param_arr,dec,decT,decD):
    
    layer_field = QgsFields()
    layer_field.append(QgsField('id', QVariant.Int))
    
    #for all routes
    for elem in route_optim_array:
        route_num = elem[0]
        #choose number of routes
        from_to_arr = from_to_arr[from_to_arr['route_num'] == route_num]
        pts_arr = []
        z_arr = np.array([])
        
        #for each from all saved routes
        for el in from_to_arr:
            pos_from = el[1]
            pos_to = el[2]
            #find route points
            pts_set = points_set(self,new_compl_poss_route_arr,pos_from,pos_to,pts_arr,z_arr)
        
        #choose attributes to append, "dec"-decimal places for distance values, "decT" - for time values, "decD" - for degree values
        attri_arr = []
        attri_arr.append(int(route_optim_array[route_num][0]))
        if length_2D:
            layer_field.append(QgsField('length_2D', QVariant.Double, 'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][2]))
        if length_3D:
            layer_field.append(QgsField('length_3D', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][3])) 
        if time:
            layer_field.append(QgsField('time', QVariant.Double,'double', 20, decT))
            attri_arr.append(float(route_optim_array[route_num][8]))
        if elevation:
            layer_field.append(QgsField('elevation', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][1]))
        if height_differences:
            layer_field.append(QgsField('height_differences', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][5]))
        if approaches:
            layer_field.append(QgsField('approaches', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][4]))
        if descents:
            layer_field.append(QgsField('descents', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][6]))
        if mean_slope:
            layer_field.append(QgsField('mean_slope', QVariant.Double,'double', 20, dec))
            attri_arr.append(float(route_optim_array[route_num][7]))
        
        #save a vector file to disk
        try:
            writer = QgsVectorFileWriter(result_file_path,'UTF-8',layer_field, QgsWkbTypes.MultiLineString, QgsCoordinateReferenceSystem("EPSG:"+str(EPSG_num)), 'ESRI Shapefile')
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to save the output file.", QMessageBox.Ok).exec()
            
        #create an empty object "QgsFeature"
        feat = QgsFeature()
        #build a geometry (multi-polyline) based on pints from "pts_set"
        feat.setGeometry(QgsGeometry.fromMultiPolylineXY([pts_set[1]]))
        #append attributes
        feat.setAttributes(attri_arr)
        #save the object as a vector file
        writer.addFeature(feat)
        #display object as a layer
        try:
            iface.addVectorLayer(result_file_path,str(route_num)+'_','ogr')
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to display the output file.", QMessageBox.Ok).exec()
        
    return None
