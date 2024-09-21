import cv2
import os
# Load the image
for loc in os.listdir('photos/dataset_final/wire_L_r2'):
    image = cv2.imread(os.path.join('photos/dataset_final/wire_L_r2',loc))

# Horizontally flip the image
    flipped_image = cv2.flip(image, 1)

# Save the flipped image
    cv2.imwrite(f'photos/dataset_final/wire_L_r3/{loc}.png', flipped_image)