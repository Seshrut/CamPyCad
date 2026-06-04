import cv2 as cv
import numpy as np
import os
import random
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import tv_tensors

individual_tranforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Lambda(lambda x: x.to(torch.device('cuda'))),
        v2.Grayscale(num_output_channels=1),
        v2.RandomRotation(degrees=(-10,10)),
        v2.Lambda((lambda x: v2.functional.resize(x,size=int(random.uniform(0.75, 1.25) * 120))))
    ])

common_transforms = v2.Compose([
        v2.RandomPerspective(distortion_scale=0.5, p=0.5),
        v2.RandomRotation(degrees=(-5,5)),
])

def get_image() -> dict[str, torch.Tensor | list]:
    """
    Generates a composite image from a dataset of labeled images and returns the image along with metadata.

    The function selects random images from a specified directory, rescales them, and arranges them in a grid.
    It also generates metadata for each image, including its position and label.

    Returns:
        dict: A dictionary containing the composite image and metadata.
            - "image": A tensor representing the composite image.
            - "boxes": A tensor of bounding box coordinates for each image in the composite.
            - "labels": A list of labels corresponding to each image in the composite.
    """
    data_dir = './photos/dataset_final'
    labels = os.listdir(data_dir)  # battery, resistor, etc
    final = []  # --->[(x,y),(x+x0,y+y0),label]
    box_list = []
    labels_list = []
    y = 0
    first_row = True
    for i in range(5):
        x = 0
        first_img = True
        for j in range(5):
            # one of labels (eg. battery)
            label = labels[random.randint(0, len(labels)-1)]
            # label = labels[random.randrange(0, 8, 3)]# for now
            labelpos = os.path.join(data_dir, label)
            # one of image of label
            img = os.listdir(labelpos)[random.randint(0, len(os.listdir(labelpos))-1)]
            img = read_image(os.path.join(labelpos, img))
            img = individual_tranforms(img)
            img = img.squeeze(0)

            # add random gap on left for each img
            rand_x_left_gap = random.randint(0, 10)
            x += rand_x_left_gap
            # add random gap on top
            rand_y_top_gap = random.randint(0, 10)
            # compensate gap at bottom to keep img consistant height
            rand_y_bottom_gap = 260-img.shape[0]-rand_y_top_gap
            y_img = y+rand_y_top_gap
            rand_x_right_gap = 260-img.shape[1]-rand_x_left_gap
            # image is now at a different y level than the row y level
            h,w = img.shape
            final.append([[x, y_img, x+h, y_img+h], label])
            box_list.append([x, y_img, x+h, y_img+h])
            labels_list.append(label)
            # print(img.shape)
            img = torch.cat((
                torch.zeros((rand_y_top_gap, w),dtype=img.dtype,device=img.device),  # add to top
                img,
                torch.zeros((rand_y_bottom_gap, w),dtype=img.dtype,device=img.device) # add to bottom
                ), dim=0)  # add to top and bottom
            img = torch.cat((
                torch.zeros((img.shape[0], rand_x_left_gap),dtype=img.dtype,device=img.device),  # add to left
                img,
                torch.zeros((img.shape[0], rand_x_right_gap),dtype=img.dtype,device=img.device) # add to right
                ), dim=1)  # add to left and right
            x += 260
            h,w = img.shape

            if first_img:
                row = img
                # row = torch.cat((torch.zeros((h, rand_x_gap),dtype=img.dtype,device=img.device), img), dim=1)  # add to left
                first_img = False
            else:
                # row += image
                row = torch.cat((row, img), dim=1)  # add to left

        # outside "for j" loop
        row_h, row_w = row.shape
        y += 260
        if first_row:
            # picture = row
            picture = row
            first_row = False
        else:
            # picture += row
            picture = torch.cat((picture, row), dim=0)
    box_list = torch.tensor(box_list, dtype=torch.float32)
    box_list = tv_tensors.BoundingBoxes(box_list, format="XYXY",canvas_size=(picture.shape[0], picture.shape[1]))
    picture = picture.unsqueeze(0)  # add channel dimension
    output = {"image": picture, "boxes": box_list, "labels": labels_list}
    output = common_transforms(output)
    output["image"] = output["image"].squeeze(0)  # remove channel dimension

    return output



if __name__ == "__main__":
    out = get_image()
    print(out["image"].shape)
    print(out["labels"])
    out["image"] = out["image"].cpu().numpy().astype(np.uint8)
    # add bounding boxes to image
    for box, label in zip(out["boxes"], out["labels"]):
        x1, y1, x2, y2 = box.int().tolist()
        cv.rectangle(out["image"], (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.putText(out["image"], label, (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv.imshow('img', out["image"])
    # close if q is pressed
    while True:
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    # cv.destroyAllWindows()