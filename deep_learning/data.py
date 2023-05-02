import torch.utils.data as data
from torchvision.transforms import ToTensor, Normalize, Compose


class ImageData(data.Dataset):
    """A PyTorch Dataset that loads image files and applies the given transforms."""
    
    def __init__(self, image_files, transforms=None):
        self.image_files = image_files
        self.transforms = transforms or Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.transforms(image)
        return image
    
    def __len__(self):
        return len(self.image_files)
