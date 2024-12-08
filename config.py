import os
import torch
import numpy as np
class Main():
    def __init__(self,dirloc:str='./photos/dataset_final'):
        self.trainingDataDir = dirloc
    def getLabelList(self)->list[str]:
        """
        Retrieves the list of labels. If the list is not already available,
        populates it by listing the contents of the training data directory.

        Returns:
            list: A list of labels.
        """
        try:
            return self.labelList
        except:
            self.labelList = os.listdir(self.trainingDataDir)
            return self.labelList
    def getLabelDict(self, reverseRelation:bool=False)->dict[int,str]:
        '''
        Returns a dictionary mapping labels to indices or indices to labels.

        If reverseRelation is False, the method returns a dictionary where the keys are the labels
        and the values are their corresponding indices. If reverseRelation is True, it returns the
        existing labelDict which maps indices to labels.

        The method attempts to create the labelDict from labelList if it doesn't exist. If labelList
        is also not available, it tries to generate it using the getLabelList method.

        Parameters:
        reverseRelation (bool): Determines the direction of the mapping. Defaults to False.

        Returns:
        dict: A dictionary mapping labels to indices or indices to labels.
        '''
        try:
            if reverseRelation:
                return {v: k for k, v in self.labelDict.items()}
            else:
                return self.labelDict
        except:  # no labelDict
            try:
                self.labelDict = dict(zip([i for i in range(len(self.labelList))],self.labelList))
                return self.getLabelDict(reverseRelation=reverseRelation)

            except:  # no labelList
                self.labelList = self.getLabelList()
                return self.getLabelDict(reverseRelation=reverseRelation)
    def formatImgData(self, x: np.ndarray, y: list) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Formats image data and associated labels into a structure suitable for model input.

        Args:
            x (np.ndarray): The image data as a NumPy array with shape (height, width, channels).
            y (list): A list of tuples, where each tuple contains bounding box coordinates and a label index.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The image data converted to a PyTorch tensor with shape (channels, height, width).
                - dict[str, torch.Tensor]: A dictionary with keys 'boxes' and 'labels', where:
                    - 'boxes' is a tensor of bounding box coordinates with dtype torch.float.
                    - 'labels' is a tensor of label indices with dtype torch.int64.
        """
        thisImgData = {}
        l_boxes = []
        l_labels = []
        for j in range(len(y)):
            l_boxes.append(y[j][0])
            l_labels.append(self.getLabelDict(reverseRelation=True)[y[j][1]])
        thisImgData['boxes'] = torch.tensor(l_boxes, dtype=torch.float)
        thisImgData['labels'] = torch.tensor(l_labels, dtype=torch.int64)
        img = torch.from_numpy(x.transpose(2, 0, 1)).to(dtype=torch.float)
        return img, thisImgData
