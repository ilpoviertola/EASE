import torch

from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, length=1_000, img_size=(224, 224), *args, **kwargs):
        self.length = length
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # imgs_tensor = torch.randn((5, 3, 512, 512))
        imgs_tensor = torch.randn((5, 3, self.img_size[0], self.img_size[1]))
        audio_tensor = torch.randn((5, 128))
        # mask_tensor = torch.randint(0, 2, (10, 1, 512, 512)).float()
        mask_tensor = torch.randint(
            0, 2, (10, 1, self.img_size[0], self.img_size[1])
        ).float()

        target = {
            "masks": mask_tensor,
            "labels": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            "mask_to_frame_idx": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
            "is_neg_sample": torch.tensor([False] * 5),
        }

        return imgs_tensor, audio_tensor, target
