import cv2
import numpy as np
import os

def draw_path_on_image(image_path, path, color=(0, 0, 255), thickness=2):
    """
    Draws a path on an image and saves it to disk.

    Args:
        image_path (str): Path to the input image.
        path (list): List of node coordinates (tuples) representing the path.
        color (tuple): Color of the path to be drawn (in BGR format).
        thickness (int): Thickness of the path line to be drawn.

    Returns:
        None.
    """
    # Load the input image
    img = cv2.imread(image_path)

    # Draw the path on the image
    for i in range(1, len(path)):
        cv2.line(img, path[i-1], path[i], color, thickness)

    # Save the result to disk
    output_dir = os.path.dirname(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0] + '_path.jpg'
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)


def visualize_path_on_map(image_path, path, path_color=(0, 0, 255), path_thickness=2,
                          map_opacity=0.5, map_color=(255, 255, 255)):
    """
    Draws a path on a map and saves it to disk.

    Args:
        image_path (str): Path to the input map image.
        path (list): List of node coordinates (tuples) representing the path.
        path_color (tuple): Color of the path to be drawn (in BGR format).
        path_thickness (int): Thickness of the path line to be drawn.
        map_opacity (float): Opacity of the input map image to be blended with the path overlay.
        map_color (tuple): Color of the map to be blended with the path overlay.

    Returns:
        None.
    """
    # Load the input image
    img = cv2.imread(image_path)

    # Create a mask for the path
    mask = np.zeros_like(img)
    for i in range(1, len(path)):
        cv2.line(mask, path[i-1], path[i], path_color, path_thickness)

    # Create a map overlay by blending the input image with a constant color
    map_overlay = np.zeros_like(img) + map_color
    map_overlay = cv2.addWeighted(img, map_opacity, map_overlay, 1 - map_opacity, 0)

    # Create a path overlay by blending the map overlay with the path mask
    path_overlay = cv2.addWeighted(map_overlay, 1, mask, 0.5, 0)

    # Save the result to disk
    output_dir = os.path.dirname(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0] + '_path.png'
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, path_overlay)
