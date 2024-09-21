import os
import cv2

# Load the image
for loc in os.listdir('photos/dataset_final/wire_st_r0'):
    image = cv2.imread(os.path.join('photos/dataset_final/wire_st_r0',loc))
# Rotate the image 90 degrees counterclockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Save the rotated image
    cv2.imwrite(f'photos/dataset_final/wire_st_r1/{loc}.png', rotated_image)