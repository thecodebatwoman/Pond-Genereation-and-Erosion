#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot
from scipy import ndimage as ndi
from skimage import morphology


# In[ ]:


def generate_pond():
    # randomly genterate a pond
    pond_seed_a = (np.random.rand(100,140)) # np.random.rand(l,w) generate an image of length l, and width w
    pond_seed_b = pond_seed_a**5 # increase range between values
    # make binary
    pond_seed_b[pond_seed_b > np.average(pond_seed_a)] = 1 # make values over the average 1
    pond_seed_b[pond_seed_b < np.average(pond_seed_a)] = 0 # make values under the average 0
    # use image filters to refine random binary image into a pond shape
    pond_c = ndi.binary_dilation(pond_seed_b)
    pond_d = ndi.binary_closing(pond_c)
    pond_e = ndi.binary_fill_holes(pond_d)
    pond_f = morphology.binary_opening(pond_e,morphology.disk(20))
    return(pond_f)


# In[ ]:


def pond_drying_over_time(image):

    new_frames = np.array([image]) # new image array with staring image to add subsequent images to

    for i in range(0,9): # do 10 times

        eroded_frame_a = np.array([ndi.binary_erosion(new_frames[i], iterations=i+1)]) # erode pond


        new_frames = np.vstack([new_frames, eroded_frame_a]) # store new eroded images in a stack along the first axis (time)

    return(new_frames)


# In[ ]:


def find_edge(image):

        ## edge detected from top down

    drop_edge_y_bot = np.argmax(image,axis=0) # y coord of where array value is highest i.e. goes from zero to one, axis=0 gives array in locations, bottom edge
    drop_edge_x_bot = np.arange(0,len(drop_edge_y_bot))  # x coord based on length of y values array

    drop_edge_y_top = np.shape(image)[0] - np.argmax(np.flip(image),axis=0) # find top edge by fliping, align with original coordinates
    drop_edge_x_top = np.flip(np.arange(0,len(drop_edge_y_top)))

    drop_edge_x_rh = np.shape(image)[1] - np.flip(np.rot90([np.argmax(np.rot90(image),axis=0)], k=3), axis=1) # find right hand side by rotation, align with original coordinates
    drop_edge_y_rh = np.arange(0,len(drop_edge_x_rh))

    drop_edge_x_lh = np.flip(np.argmax(np.rot90(image, k=3),axis=0))  # find left hand side by rotation, align with original coordinates
    drop_edge_y_lh = np.arange(0,len(drop_edge_x_rh))

    drop_edge_y_1 = np.append(drop_edge_y_bot,drop_edge_y_top) # store y edges into one array
    drop_edge_y_2 = np.append(drop_edge_y_1,drop_edge_y_rh)
    drop_edge_y = np.append(drop_edge_y_2,drop_edge_y_lh)

    drop_edge_x_1 = np.append(drop_edge_x_bot,drop_edge_x_top) # store x edges into one array
    drop_edge_x_2 = np.append(drop_edge_x_1,drop_edge_x_rh)
    drop_edge_x = np.append(drop_edge_x_2,drop_edge_x_lh)


    corner_coord_l = np.where(drop_edge_x == 0) # remove cells coresponding with image boarders
    corner_coord_t = np.where(drop_edge_y == 0)
    corner_coord_b = np.where(drop_edge_y == np.shape(image)[0])
    corner_coord_r = np.where(drop_edge_x == np.shape(image)[1])

    corner_coord_1 = np.append(corner_coord_l, corner_coord_t)
    corner_coord_2 = np.append(corner_coord_1, corner_coord_b)
    corner_coord = np.append(corner_coord_2, corner_coord_r)

    drop_edge_x_d = np.delete(drop_edge_x, corner_coord)
    drop_edge_y_d = np.delete(drop_edge_y, corner_coord)

    return(drop_edge_x_d,drop_edge_y_d)


# In[ ]:


def edge_stack(image_stack):

    drop_edge_xs = [] # create empty array to store data for each iteration
    drop_edge_ys = []


    for i in range(0,image_stack.shape[0]):
        drop_edge_x,drop_edge_y = find_edge(image_stack[i])

        drop_edge_xs.append(drop_edge_x) # add new data slice to original array
        drop_edge_ys.append(drop_edge_y)

    return(drop_edge_xs, drop_edge_ys)


# In[ ]:


def plot_contour_stack(image_stack):

        ## plot overlaping droplet edges over time
    pyplot.figure(figsize=(7, 5))
    ax = pyplot.gca()
    pyplot.title('Pond Erosion Over Time')
    pyplot.xticks([])
    pyplot.yticks([])
    for i in range(0,image_stack.shape[0],1):
        ## (start at 0, length of time array, step count), adjust step count for plot density

        drop_edge_x,drop_edge_y = find_edge(image_stack[i])

            ## blue
        pyplot.plot(drop_edge_x,drop_edge_y, linewidth=0.8, marker='.', ls='', c=( 0, ((i/image_stack.shape[0])),1))


# In[ ]:


pond = generate_pond()


# In[ ]:


pyplot.imshow(pond, vmin = 0, vmax = 2, cmap = 'ocean')
pyplot.title('Randomly Generated Pond')


# In[ ]:


new_stack = pond_drying_over_time(pond)


# In[ ]:


drop_edge_xs, drop_edge_ys = edge_stack(new_stack)


# In[ ]:


j=4 # select a frame to view
pyplot.imshow(new_stack[j], alpha = 0.5, vmin = 0, vmax = 2, cmap = 'ocean')

pyplot.plot(drop_edge_xs[j],drop_edge_ys[j], marker='.', ls='', c='k')
pyplot.title('Edges of Randomly Generated Pond')


# In[ ]:


plot_contour_stack(new_stack)


# In[ ]:




