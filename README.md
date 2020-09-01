# PracaMagisterska_AK

## ABOUT
A QGIS plugin to find the best route between points. 
Takes an elevation from the grid dataset and assigns it to the (2D) linear layer, 
basing on points layer and given conditions return an optimized route with suitable attributes e.g. distance, time.

## MAIN REQUIREMENTS
* raster in the form of a rectangle
* information about height contained in Band (1)
* the linear layer must not go beyond the raster range
* layers in a flat, rectangular coordinate system
* developed for QGIS 3.12

## START-UP
Copy "findroute" catalog and paste into -> QGIS (version)\apps\qgis\python\plugins.
Open QGIS, open "Find Route" (-> "Find the best route between points") from the "Plugins" bookmark. 
To update code changes you can download "Plugin reloader".

## METHOD OF WORKING
1. Choose grid (with height values), linear layer (roads, paths) and points layer (with the course of route). 
2. Choose type of optimization (Minimization/Maximization), optimization condition and method of movement. 
   If you want to have a constant veliocity, then type an additional value (in km/h), which is optional.
3. Select a set of attributes, which you want to add to the output route layer. "Id" value will allways be generated.
4. Your output file will be saved in a given location and display in a current project.

To test this plugin you can use data from the "TestData" folder.

![GitHub Logo](/pictures/1.jpg)



