# -*- coding: utf-8 -*-
"""
/***************************************************************************
 FindRoute
                                 A QGIS plugin
 Find the best route between points
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2020-05-20
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Karolina Aksamit
        email                : karolina.aksamit.ak@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .findRoute_dialog import FindRouteDialog
import os.path


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

#_____IMPORT_EXTERNAL_FUNCTIONS_____
from .functions import open_raster, raster_height_array, raster_parameters, raster_XYZ_array,\
open_vector, vector_objects, vector_objects_elements, empty_array, vertex_information_array,\
pixel_elevation, add_elevation, create_type_array, sort_lines_elements, add_array_element,\
raster_line_intersection, new_vertex_elevation_array, new_vertex_information_array,\
points_objects_elements, points_information_array, closest_point, closest_pts_possibility,\
delete_arr_elem, next_origin_vertex, prev_origin_vertex, check_elem_exist, duplicate_vertex,\
add_next_elem_array, add_equal_id, other_objects, calc_last_vert, calc_last_vert_prev,\
add_type_element_arr, next_vert2, prev_vert2, identical_line, calc_routes, possible_routes,\
complete_possible_route, check_array_shortages, add_type_element_section, calculate_time,\
calculate_slope_vrt_position, route_sections_attributes, route_optimization, points_set,\
create_shp_route

# ###############FUNCTIONS################

class FindRoute:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'FindRoute_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Find Route')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('FindRoute', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/findRoute/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Find the best route between points'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Find Route'),
                action)
            self.iface.removeToolBarIcon(action)

#______________________________FUNCTIONS__________________________________
#_________________________________________________________________________
 
#_____LAYERS_____
    
    #check layer extension for all "layers" from project and return an array 
    #...wich contains names of all layers with the specified extension "ext"
    def load_layers_to_frame(self,layers,ext):
        #save layers with appropriate extension
        arr_name = []
        for la in layers:
            #get the path
            layer_path = la.layer().dataProvider().dataSourceUri()
            layer_directory, layer_name = os.path.split(layer_path)
            #get the layer name with its extension
            try:
                layer_name, rest = layer_name.split("|")
            except:
                layer_name = layer_name[:]
            #check if layer ends with appropriate extension
            if layer_name.endswith(ext):
                arr_name.append(layer_name)
        
        return arr_name

    
    #select place to save an output
    def select_output_file(self):
        filename, _filter = QFileDialog.getSaveFileName(self.dlg, "Select output file ","", '*.shp')
        self.dlg.outputLE.setText(filename)
    
    
    #select files from browser: 1)grid, 2)linear layer and 3)points layer
    def select_input_file1(self):
        filename, filter = QFileDialog.getOpenFileName(self.dlg,"Open File","","*.tif")
        if filename != "":
            self.dlg.inputGridCB.addItems([filename])


    def select_input_file2(self):
        filename, filter = QFileDialog.getOpenFileName(self.dlg,"Open File","","*.shp")
        if filename != "":
            self.dlg.inputLinesCB.addItems([filename])


    def select_input_file3(self):
        filename, filter = QFileDialog.getOpenFileName(self.dlg,"Open File","","*.shp")
        if filename != "":
            self.dlg.inputPointsCB.addItems([filename])
    
    
    #get the path, if path not exists -> pass
    def get_path(self,layers,box_text):
        for la in layers:
            try:
                layer_path = la.layer().dataProvider().dataSourceUri().split("|")
                if layer_path[0].find(box_text) != -1:
                    raster_file_path = layer_path[0]
            except:
                pass
        return str(raster_file_path)
     
    
    #read paths to files: grid, linear layer, points layer
    def read_input_file(self,layers):
    
        #take current text from the "inputGridCB", "inputLinesCB", "inputPointsCB"
        raster_file_path = self.dlg.inputGridCB.currentText()
        lines_file_path = self.dlg.inputLinesCB.currentText() 
        points_file_path = self.dlg.inputPointsCB.currentText()
        result_file_path = self.dlg.outputLE.text()


        #check if the text is a path to the file, if not -> find a path to the file, if it is impossible -> error
        try:
            if not os.path.isfile(raster_file_path):
                raster_file_path = self.get_path(layers,raster_file_path)
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to find the grid layer path.\nThe file does not exist or try refresh the plugin.", QMessageBox.Ok).exec()
        try:
            if not os.path.isfile(lines_file_path):
                lines_file_path = self.get_path(layers,lines_file_path)
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to find the linear layer path.\nThe file does not exist or try refresh the plugin.", QMessageBox.Ok).exec()
        try:
            if not os.path.isfile(points_file_path):
                points_file_path = self.get_path(layers,points_file_path)
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to find the points layer path.\nThe file does not exist or try refresh the plugin.", QMessageBox.Ok).exec()
        
        #check if the text is a right path, if not -> error
        if result_file_path.find("\0"):
            result_file_path = result_file_path.replace("\0","\\0")
            dirname = os.path.dirname(result_file_path) or os.getcwd()    
        try:
            dirname = os.path.dirname(result_file_path) or os.getcwd()
            creatable = os.access(dirname, os.W_OK) 
        except:
            QMessageBox(QMessageBox.Warning, "Error","Failed to find the output file path.", QMessageBox.Ok).exec()
        
        return raster_file_path, lines_file_path, points_file_path, result_file_path
    

#_____CONDITIONS_____

    #check selected optimization condition
    def check_optimization(self):
        optimization = "none"
        if self.dlg.lengthRB.isChecked():
            optimization = 'length_3D'
        if self.dlg.timeRB.isChecked():
            optimization = 'time'
        if self.dlg.heightDifRB.isChecked():
            optimization = 'height_differences'
        if self.dlg.approachesRB.isChecked():
            optimization = 'approaches'
        if self.dlg.descentsRB.isChecked():
            optimization = 'descents'
        if self.dlg.slopesRB.isChecked():
            optimization = 'mean_slope'
        return optimization
    
    
    #check optimization option: minimalization/maximization
    def check_min_max(self):
        if self.dlg.minMaxCB.currentText() == "min":
            cond = "min"
        elif self.dlg.minMaxCB.currentText() == "max":
            cond = "max"
        return cond


#_____ATTRIBUTES_____
    
    #choose attributes to attach
    def choose_attributes(self):
        length_2D = self.dlg.length2DChB.checkState()
        length_3D = self.dlg.length3DChB.checkState()
        time = self.dlg.timeChB.checkState()
        elevation = self.dlg.elevationChB.checkState()
        height_differences = self.dlg.heightDifChB.checkState()
        approaches = self.dlg.approachesChB.checkState()
        descents = self.dlg.descentsChB.checkState()
        mean_slope = self.dlg.slopeChB.checkState()
        return length_2D, length_3D, time, elevation, height_differences, approaches, descents, mean_slope


#_____MAIN_WINDOW_____
    
    #close main window, when "closePB" clicked
    def close_main_window(self):
        self.dlg.accept()
    
    #run all application processes, when "runPB" clicked
    def run_all(self,layers):
        
        #[1] get paths
        all_paths = self.read_input_file(layers)
        raster_file_path = all_paths[0]
        lines_file_path = all_paths[1]
        points_file_path = all_paths[2]
        result_file_path = all_paths[3]
         
        #[2] open a raster file
        raster_file = open_raster(self,raster_file_path)
        
        #[3] create heights array from the grid
        raster_height_arr = raster_height_array(self,raster_file)
        
        #[4] return raster parameters
        raster_param_arr = raster_parameters(self,raster_file,raster_height_arr)

        #[5] return x,y,z array for all raster pixels
        XYZ = raster_XYZ_array(self,raster_param_arr[3],raster_param_arr[4],raster_param_arr[5],raster_param_arr[6],raster_param_arr[7],raster_param_arr[8],raster_height_arr)
        
        #[6] open the file a the linear layer
        lines = open_vector(self,lines_file_path)

        #[7] return objects from liear layer
        fts = vector_objects(self,lines)
        
        #[8] save information about existing vertex into an array
        vertex_array = vertex_information_array(self,lines,fts)
        
        #[9] find an intersection between pixels and linear layer, finally return an array: [object number, first vertex value, next vertex value, new vertex number, x, y]
        end_std_new_vert_arr = raster_line_intersection(self,lines,raster_param_arr[4],raster_param_arr[3],raster_param_arr[5],raster_param_arr[6],raster_param_arr[7],raster_param_arr[8],vertex_array)
        
        #[10] assign heights to all vertices
        new_z_arr = new_vertex_elevation_array(self,end_std_new_vert_arr,XYZ,raster_param_arr)

        #[11]return an array with sorted information about vertices
        new_info_arr = new_vertex_information_array(self,end_std_new_vert_arr,new_z_arr,4,4)

        #[12] open a file with the points layer
        route_pts = open_vector(self,points_file_path)

        #[13] return an array with information about point layer
        route_pts_arr = points_information_array(self,route_pts)
        
        #[14] find the closest vertices to points 
        closest_pts_arr0 = closest_point(self,route_pts_arr, new_info_arr)
        
        #[15] return an array with the possible combinations of routes between the closest vertices
        closest_pts_arr = closest_pts_possibility(self,closest_pts_arr0,new_info_arr)

        #[16] find all possible routes
        possible_routes_arr =  possible_routes(self,closest_pts_arr,new_info_arr)
        
        #[17] complete route array
        compl_poss_route_arr = complete_possible_route(self,possible_routes_arr,new_info_arr)
        
        #[18] fill shortages on "compl_poss_route_arr"
        new_compl_poss_route_arr = check_array_shortages(self,compl_poss_route_arr,possible_routes_arr)       
        
        #[19] make a table with attributes for sections of the routes (routes between two points)
        sections_attr_arr = route_sections_attributes(self,new_compl_poss_route_arr)
        
        #[20] choose the best route using optimization parameters 
        route_optim_array = route_optimization(self,sections_attr_arr,closest_pts_arr0)
        
        #[21] choose atributes to be attached
        atr = self.choose_attributes()
        
        #[22] create and display an output file
        create_shp_route(self,atr[0],atr[1],atr[2],atr[3],atr[4],atr[5],atr[6],atr[7],result_file_path,raster_param_arr[11],route_optim_array[0],new_compl_poss_route_arr,route_optim_array[1],raster_param_arr,3,2,2)
                     
        return  None

#_________________________________________________________________________


    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = FindRouteDialog()


#_______________________________LAYERS____________________________________
#_________________________________________________________________________
        
        
            # Link to the currently loaded layers from the project
            layers = QgsProject.instance().layerTreeRoot().children()
            
            
            #_____Add grid layer_____
            
            # Clear the contents of the inputGridCB from previous runs
            self.dlg.inputGridCB.clear()
            # Connect the inputGridCB with names of the specified layers
            self.dlg.inputGridCB.addItems(self.load_layers_to_frame(layers,"tif"))

            #Open a file browser
            self.dlg.openGridTB.clicked.connect(self.select_input_file1)


            #_____Add linear layer_____
            self.dlg.inputLinesCB.clear()
            self.dlg.inputLinesCB.addItems(self.load_layers_to_frame(layers,"shp"))
            
            self.dlg.openLinesTB.clicked.connect(self.select_input_file2)
            
            
            #_____Add points layer_____
            self.dlg.inputPointsCB.clear()
            self.dlg.inputPointsCB.addItems(self.load_layers_to_frame(layers,"shp"))
            
            self.dlg.openPointsTB.clicked.connect(self.select_input_file3)
            
            
            #_____Output file path_____
            self.dlg.openOutputTB.clicked.connect(self.select_output_file)


#_______________________________CONDITIONS________________________________
#_________________________________________________________________________


#_______________________________ATTRIBUTES________________________________
#_________________________________________________________________________


#______________________________MAIN_WINDOW________________________________
#_________________________________________________________________________
           
           #_____CLOSE_____
            self.dlg.closePB.clicked.connect(self.close_main_window)
            
            #_____RUN_____
            self.dlg.runPB.clicked.connect(lambda: self.run_all(layers))

        
        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            # Do something useful here - delete the line containing pass and
            # substitute with your code.
            pass
