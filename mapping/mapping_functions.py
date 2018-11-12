import numpy as np

'''
Project from coordinates in map xyz coordinates back onto original image

xyz_raw: the map coordinates
pmatrix: the pmatrix from pix4d for the given image points are projecting to
offset: from pix4d
'''
def get_uv(xyz_raw, pmatrix, offset):
    xyz = xyz_raw - offset
    xyz1 = np.ones((4, 1))
    xyz1[:3] = np.expand_dims(xyz, 1)
    xyz_image = np.matmul(pmatrix, xyz1)
    uv = np.array([xyz_image[0] / xyz_image[2], xyz_image[1] / xyz_image[2]])
    return np.squeeze(uv)

'''
Project from uv image coordinates to map xyz coordinates for some given mu

uv: image coordinates
inv_m: inverse m matrix (m matrix is the left three columns of pmatrix)
p4: the fourth colummn of the pmtatix
mu: a term that chooses a point along projection ray
'''
def get_xyz_proj(uv, inv_m, p4, mu, offset):
    uv1 = np.ones(3)
    uv1[:2] = uv 
    xyz = np.matmul(inv_m, ((uv1 * mu) - p4))
    xyz += offset
    return xyz

'''
Transforms from utm map coordinates to raster map coordinates

x_utm: x coordinate in utm
y_utm: y coordinate in utm
x_origin: of raster image in utm 
y_origin: of raster image in utm
pixel_height: of raster image in utm
pixel_width: of raster image in utm
image_scale: if the raster image being used has been scaled from the original image
             image scale of .5 means that the map is being used at
             .5 w .5 h compared to original

'''
def utm_to_raster(x_utm, y_utm, x_origin, y_origin, pixel_width, pixel_height, image_scale):
    x_raster = int(((x_utm - x_origin) / pixel_width) * image_scale)  
    y_raster = int(((y_utm - y_origin) / pixel_height) * image_scale)
    return((x_raster, y_raster))


'''
opposite of utm_to_raster
'''
def raster_to_utm(x_raster, y_raster, x_origin, y_origin, pixel_width, pixel_height):
    x_utm = ((x_raster * pixel_width) + x_origin)
    y_utm = (y_raster * pixel_height) + y_origin
    return((x_utm, y_utm))

'''
returns frame index.  Is equivelent to index for that frame in position matrix

frame_num: the raw frame num in video (60fps including takeoff etc.)
frame_names: list of frames being used for tracking (30fps, just observation)
first_frame: first frame in video of tracked observation

'''
def get_image_ind_from_frame_num(frame_num, frame_names, first_frame):
#     if frame_num % 2 != 0:
#         frame_num -= 1
    frame_ind = int((frame_num - first_frame) // 2)
    frame_file = frame_names[frame_ind]
    
    return [frame_ind, frame_file]

'''
The project image coordinate on to map as the intersection between the 
projection ray from camera throuhg image point and the ground.  Iteractively
searches for this point

uv: point in image
z_guess: where to start looking for ground
pmatrix_dict: must_contain inv_mmatrix (inverse of the first three columns of pmatrix)
              and p4 (fourth column of pmatrix)  
offset: from pix4d
elevation_map: elevation raster plot of map area
max_guesses: how many iterations to search along projection ray before returning estimate
correct_threshold: if the distance between the point on the projection ray and the ground is 
                   within this threshold stop seraching and return point
pixel_width: pixel width of pixels in elevation raster in utm units
pixel_height: like pixel width
x_origin: origin of elevation raster in utm units
y_origin: like x_origin

returns: xyz coordinates in utm
'''

def from_image_to_map(uv, z_guess, pmatrix_dict, offset, elevation_map, 
    max_guesses, correct_threshold, pixel_width, pixel_height, x_origin, y_origin):

    last_pos_guess = 0
    last_neg_guess = None
    found_point = False
    first_search = True
    guess_count = 0
    animal_height = 1
    while not found_point:
        xyz = get_xyz_proj(uv, pmatrix_dict['inv_mmatrix'], pmatrix_dict['p4'], z_guess, offset)
        x_rast, y_rast = utm_to_raster(xyz[0], xyz[1], x_origin, y_origin, pixel_width, pixel_height, image_scale=1.0)
        x_rast = min(x_rast, elevation_map.shape[1]-1)
        x_rast = max(x_rast, 0)
        y_rast = min(y_rast, elevation_map.shape[0]-1)
        y_rast = max(y_rast, 0)
        z_diff = xyz[2] + animal_height - elevation_map[y_rast, x_rast]

        if guess_count > max_guesses:
            break
        if np.abs(z_diff) <= correct_threshold:
            found_point = True
            break
        if first_search and z_diff > 0:
            last_pos_guess = z_guess
            z_guess += z_guess
        elif first_search and z_diff < 0:
            first_search = False
            new_guess = (last_pos_guess + z_guess) / 2
            last_neg_guess = z_guess
            z_guess = new_guess
        elif z_diff > 0:
            new_guess = (last_neg_guess + z_guess) / 2
            last_pos_guess = z_guess
            z_guess = new_guess
        elif z_diff < 0:
            new_guess = (last_pos_guess + z_guess) / 2
            last_neg_guess = z_guess
            z_guess = new_guess
        guess_count += 1
        
    x_utm, y_utm = raster_to_utm(x_rast, y_rast, x_origin, y_origin, pixel_width, pixel_height)
        
    return (x_utm, y_utm, found_point, z_guess)

'''
Takes a file generated by pix4d with all of the calculated pmatrix values
generates a dictionary that contains:
image_name: image_name of image that corresponds with the given pmatrix
pmatrix: numpy pmatrix
inv_mmatrix: inverse of first three columns of pmatrix
p4: last column if pmatrix

pmatrix_file: path to file generated by pix4d

returns list of dictioanries 
'''

def create_pmatrix_dicts(pmatrix_file):
    with open(pmatrix_file) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    pmatrix_list = []
    for line in content:
        split_line = line.split(' ')
        pmatrix = np.zeros((3, 4))
        for row in range(pmatrix.shape[0]):
            pmatrix[row, :] = split_line[(1 + row * 4):(1 + (row+1) * 4)]
        mmatrix = np.copy(pmatrix[:, :3])
        inv_mmatrix = np.linalg.pinv(mmatrix)
        p4 = np.copy(pmatrix[:, 3])
        pmatrix_list.append({'image_name': split_line[0], 'pmatrix': pmatrix, 'inv_mmatrix': inv_mmatrix, 'p4': p4})
    pmatrix_list.sort(key=lambda pmatrix_dict: 
        (int(pmatrix_dict['image_name'].split('_')[-2]), int(pmatrix_dict['image_name'].split('.')[-2].split('_')[-1]))) 

    return pmatrix_list