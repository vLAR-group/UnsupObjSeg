
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, distance_transform_edt

def convex_hull(binary_mask):
    """Compute the convex hull of a binary mask."""
    points = np.argwhere(binary_mask)
    if len(points) < 3:
        return binary_mask.copy()  # Not enough points to form a convex hull
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    return hull_mask

def find_deepest_concavity(distance_map):
    """Find the deepest concavity in the distance transform."""
    return np.unravel_index(np.argmax(distance_map), distance_map.shape)

def cut_region(binary_mask, cut_point, direction):
    """Cut the region based on the cut_point and direction, returning the smaller part."""
    y, x = cut_point
    height, width = binary_mask.shape

    # Create a mask for the entire region
    mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Define the slope from the direction
    dy, dx = np.sin(direction), np.cos(direction)

    # Create a half-plane mask
    for i in range(height):
        for j in range(width):
            # Calculate the position relative to the cut point
            if (dy * (j - x) - dx * (i - y)) > 0:  # Above the line
                mask[i, j] = 1

    # The smaller part of the original object is where the mask overlaps with the binary mask
    smaller_part = mask & binary_mask

    return smaller_part

def maximal_inscribed_convex_set(binary_mask):
    """Compute the maximal inscribed convex set."""
    # Fill holes in the binary mask
    # filled_mask = binary_fill_holes(binary_mask).astype(np.uint8)
    filled_mask = binary_mask.astype(np.uint8)

    # Compute the convex hull
    hull = convex_hull(filled_mask)

    # Compute convex deficiency D
    deficiency = hull - filled_mask

    while True:
        # Calculate the distance transform of the deficiency
        distance_map = distance_transform_edt(deficiency)

        # Find the deepest concavity
        deepest_concavity = find_deepest_concavity(distance_map)
        print('deepest_concavity', deepest_concavity, distance_map[deepest_concavity])

        # Check if the deepest concavity is within acceptable bounds
        if distance_map[deepest_concavity] <= 3:
            break

        # Generate cuts in 8 directions
        cuts = [cut_region(filled_mask, deepest_concavity, n * np.pi / 4) for n in range(8)]
        
        # Evaluate the size of the resulting sub-regions
        sub_regions = [filled_mask & cut for cut in cuts]
        areas = [np.sum(region) for region in sub_regions]

        # Remove the smallest sub-region if its area is greater than 0
        valid_areas = [area for area in areas if area > 0]
        print('valid_areas', valid_areas)
        if not valid_areas:
            break  # Exit if no valid areas are found

        min_area_index = np.argmin(valid_areas)
        filled_mask = filled_mask & ~sub_regions[min_area_index]

        # Recompute the convex hull and deficiency
        hull = convex_hull(filled_mask)
        deficiency = hull - filled_mask

    return filled_mask

# Example usage
if __name__ == "__main__":
    # Create a sample binary mask (connected region)
    region = np.zeros((1000, 1000))
    cv2.rectangle(region, (30, 30), (600, 600), 1, -1)
    region = cv2.rectangle(region, (30, 30), (400, 400), 0, -1)  # Create an overlapping rectangle

    # Compute the maximal inscribed convex set
    convex_set = maximal_inscribed_convex_set(region)

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Region")
    plt.imshow(region, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Maximal Inscribed Convex Set")
    plt.imshow(convex_set, cmap='gray')

    plt.show()
