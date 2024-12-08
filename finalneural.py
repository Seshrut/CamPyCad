import torchvision
from imgsticher import get_image
import time
from config import Main
import torch

mainobj = Main()

data_dir = mainobj.trainingDataDir
lable_Dict = mainobj.getLabelDict()
Dict_lable = mainobj.getLabelDict(reverseRelation=True)
batch_size = 100
num_batches = 100
TTsum = 0
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.train()
for batch in range(1,num_batches+1):
    images = []
    img_data = []
    Tsum = 0
    for i in range(1,batch_size+1):
        s = time.time()
        x, y = get_image()
        e = time.time()
        Tsum += (e - s)
        print(f'Batch {batch}/{num_batches}, Image {str(i) if i>9 else '0'+str(i)}/{batch_size}, Time {round(Tsum,4)}s', end='\r')
        img,imgData = mainobj.formatImgData(x,y)
        images.append(img)
        img_data.append(imgData)
    TTsum+=Tsum
    print(f'Batch {batch}/{num_batches}, Image {str(i) if i>9 else '0'+str(i)}/{batch_size}, Time {round(Tsum,4)}s')
    del images, img_data, img, imgData
    model(images, img_data)
    # Clear the batch from memory
print(f'{TTsum} secs for {batch_size * num_batches} images')

# Save the model
torch.save(model, 'mod.pth')
input('press enter to exit')
