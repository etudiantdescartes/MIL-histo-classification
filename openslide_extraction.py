import openslide
from openslide.deepzoom import DeepZoomGenerator
from skimage.color import rgb2hsv
import numpy as np
import os
from util import mkdir
from PIL import ImageDraw
from glob import glob
from multiprocessing import Pool
import argparse



def draw_grid(wsi, rect_list, extraction_level, grid_path):
    """
    Draw a grid showing the extracted tiles, on the level 3 of the WSI (the highest resolution that my pc can handle)
    Saving image as a PNG
    """
    #Level of the WSI to draw the grid on
    thumbnail_level = 3
    
    #Scaling the coordinates of the extracted tiles to the thumbnail level, to draw the grid
    diff = abs(thumbnail_level - extraction_level)
    if thumbnail_level > extraction_level:
        for rect in rect_list:
            for i in range(len(rect)):
                rect[i] = rect[i] / (diff * 2)
    elif thumbnail_level < extraction_level:
        for rect in rect_list:
            for i in range(len(rect)):
                rect[i] = rect[i] * (diff * 2)

    #Drawing rectangles
    grid_thumbnail = wsi.read_region(location=(0,0), level=thumbnail_level, size=(wsi.level_dimensions[thumbnail_level]))
    draw = ImageDraw.Draw(grid_thumbnail)
    for rect in rect_list:
        draw.rectangle(rect, outline='green', width=5)#10
    
    grid_thumbnail.save(grid_path)


def is_tissue(tile, check_area=True, tissue_area=0.5):
    """
    Changes the color space for the current tile
    Checks if there is tissue in the tile
    Checks covered area ratio by some tissue if check_area is True
    """
    rgb = np.array(tile)
    hsv = rgb2hsv(rgb)
    
    #Scale the values from 0-1 to int
    hsv[:,:,0] *= 179
    hsv[:,:,1] *= 255
    hsv[:,:,2] *= 255
    hsv = hsv.astype(np.uint8)
    
    #Thresholds (we're only applying it on the saturation channel)
    lower = np.array([0,20,0])
    upper = np.array([179, 255, 255])
    
    mask = ((lower[0] <= hsv[:,:,0]) & (hsv[:,:,0] <= upper[0]) &
            (lower[1] <= hsv[:,:,1]) & (hsv[:,:,1] <= upper[1]) &
            (lower[2] <= hsv[:,:,2]) & (hsv[:,:,2] <= upper[2]))
    
    if check_area:
        if np.sum(mask) >= mask.size*tissue_area:
            return True
    elif mask.any():
        return True
    
    return False
    
    
def extract_tiles(wsi_path, tile_size, extraction_levels, destination, output_grid):
    try:
        wsi_name = os.path.basename(wsi_path).split('.')[0]
        wsi_folder = os.path.join(destination, wsi_name)
        mkdir(wsi_folder)
        
        wsi = openslide.open_slide(wsi_path)
        dzg = DeepZoomGenerator(osr=wsi, tile_size=tile_size, overlap=0, limit_bounds=False)
        
        level_tiles = dzg.level_tiles
        
        #print(dzg.level_tiles)
        #print(dzg.level_dimensions)
        #print(wsi.level_dimensions)
        #print(wsi.level_downsamples)
        
        wsi_level_dimensions = wsi.level_dimensions
        dzg_level_dimensions = dzg.level_dimensions
        
        for extraction_level in extraction_levels:
            
            level_folder = os.path.join(wsi_folder, f'level{extraction_level}')
            mkdir(level_folder)
            wsi_level_dimension = wsi_level_dimensions[extraction_level]
    
            deepzoom_level = dzg_level_dimensions.index(wsi_level_dimension)
    
            tiles_x, tiles_y = level_tiles[deepzoom_level]
            
            rect_list = []
            for y in range(tiles_y):
                for x in range(tiles_x):
                    tile = dzg.get_tile(deepzoom_level, (x, y))
                    if is_tissue(tile, check_area=True, tissue_area=0.6):
                        xmin, ymin = x*tile_size, y*tile_size
                        xmax, ymax = xmin+tile_size, ymin+tile_size
                        rect_list.append([xmin, ymin, xmax, ymax])
                        name = f'{wsi_name}_level{extraction_level}_{xmin}_{ymin}_{xmax}_{ymax}.png'
                        tile_path = os.path.join(level_folder, name)
                        tile.save(tile_path)
            
            if output_grid:
                grid_path = os.path.join(wsi_folder, f'grid_level_{extraction_level}.png')
                draw_grid(wsi, rect_list, extraction_level, grid_path)
    except:
        print(f'Error on WSI {wsi_name}')
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='source')
    parser.add_argument('-d', dest='destination')
    parser.add_argument('-t', dest='tile_size', default=224)
    parser.add_argument('-p', dest='num_processes', default=4)
    params = parser.parse_args()
    
    wsi_paths = glob(os.path.join(params.source, '*.tif'))
    extraction_levels = sorted([1])
    output_grid = True
    tile_size = int(params.tile_size)
    arg_list = [(wsi_path, tile_size, extraction_levels, params.destination, output_grid) for wsi_path in wsi_paths]

    num_processes = int(params.num_processes)
    print('Extraction tiles...')
    with Pool(num_processes) as p:
        p.starmap(extract_tiles, arg_list)
        
    """
    #Iterative processing for debugging
    for wsi_path in wsi_paths:
        extract_tiles(wsi_path, tile_size, extraction_levels, params.destination, output_grid)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    