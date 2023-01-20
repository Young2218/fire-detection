from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CNN_Dataset(Dataset):
    def __init__(self, img_paths, labels, img_size, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(img_size,img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()                
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.transforms(image=image)['image'] / 255.
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.img_paths)