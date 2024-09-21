import pygame
wire_type = "wire_st_r0"

# Initialize Pygame
pygame.init()

# Set screen dimensions and colors
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Curve Drawing")
white = (255, 255, 255)
black = (0, 0, 0)

# Initialize drawing variables
drawing = False
start_pos = None
end_pos = None
points = []
phot_num = 0
# Main loop

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            start_pos = pygame.mouse.get_pos()
            drawing = True

        if event.type == pygame.MOUSEMOTION and drawing:
            end_pos = pygame.mouse.get_pos()
            points.append(end_pos)

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            # Save the drawn line as a PNG image
            phot_num += 1
            img_resized = pygame.transform.scale(screen,(120,120))
            pygame.image.save(img_resized, f'photos/dataset_final/{wire_type}/{phot_num}.png')
            points = []
            print(phot_num)
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE:
            points = []
    # Fill the screen with white color
    screen.fill(black)

    # Draw the line based on the points
    if len(points) > 1:
        pygame.draw.lines(screen, white, False, points, 10)

    # Update the display
    pygame.display.update()