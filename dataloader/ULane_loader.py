from PIL import Image
from glob import glob
import numpy as np

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

class CULaneDataset(Dataset):
    def __init__(self, ims_path_lst, labs_path_lst, transform=None, lab_transform=None):
        # print(ims_path_lst)
        # print(labs_path_lst)
        lab_ext = labs_path_lst[0].split('.')[-1]
        self.precompute_mask = bool(lab_ext == 'jpg')
        self.ims_path_lst = ims_path_lst
        self.labs_path_lst = labs_path_lst
        self.transform = transform
        self.lab_transform = lab_transform

    def __len__(self):
        return len(self.ims_path_lst)

    def _parse_cords(self, filepath):
        # Since there're multiple lanes per image (cubic lane, center lane, ..., etc)
        lanes_cord = []
        str2int = lambda s: int(float(s))
        with open(filepath, 'r') as f_ptr:
            for cord_lin in f_ptr.readlines():
                xpt, ypt = [], []

                # parse patch : there have some error label (negative cords), setup drop_flag
                # https://github.com/cardwing/Codes-for-Lane-Detection/issues/15#issuecomment-445170552
                drp_flg  = False

                for idx, pt in enumerate(cord_lin.split()):
                    pt = str2int(pt)
                    if pt < 0:  # drop negative x coordinate
                        drp_flg = True
                        continue
                    if drp_flg: # drop corresponding y coordinate
                        drp_flg = False
                        continue

                    # short-circuit logic : [a, b, c, d, ...] -> [ (x=a, y=b), (x=c, y=d), ... ] 
                    idx%2 == 0 and xpt.append(pt) 
                    idx%2 == 1 and ypt.append(pt)
                
                cord_pairs = list( zip(xpt, ypt) )
                lanes_cord.append(cord_pairs)

        return lanes_cord

    def _cord2mask(self, lanes_cord, mask_shape):
        mask = np.zeros(mask_shape, np.uint8)
        for lane_cord in lanes_cord:
            for idx, from_pt in enumerate(lane_cord): 
                if idx == len(lane_cord) -1:
                    break
                to_pt = lane_cord[idx + 1]
                cv2.line(mask, from_pt, to_pt, [255, 255, 255], 5) 

        return mask

    def __getitem__(self, index):

        im_path = self.ims_path_lst[index]
        image = Image.open(im_path).convert('RGB')
        #breakpoint()
        lab_path = self.labs_path_lst[index]
        if self.precompute_mask:
            image_file = Image.open(lab_path).convert('L')
            # Threshold
            lane_mask = image_file.point(lambda p: 255 if p > 124 else 0).convert('1')
            # lane_mask = torch.nn.functional.one_hot(lane_mask.long(), 2)
            # lane_mask = lane_mask.view([2, image.shape[1], image.shape[2]])

        else:  # activate runtime mask computation 
            W, H = image.size
            lanes_cord = self._parse_cords(lab_path)
            lane_mask = self._cord2mask(lanes_cord, mask_shape=[H, W, 3])

        if self.transform:
            image = self.transform(image)
        if self.lab_transform:
            lane_mask = self.lab_transform(lane_mask)
        lane_mask = torch.cat([lane_mask, 1-lane_mask], dim = 0)
        #breakpoint()
        return image, lane_mask


class CUL_datamodule(pl.LightningDataModule):
    default_trfs = transforms.Compose([ transforms.ToTensor() ])

    # Since CULane dataset use jpg, txt as im, labe format, we just hard-code
    def __init__(self, data_dir: str = '.', lab_ext: str = 'txt', val_ratio: float = 0.2, data_ld_args: dict = None):
        super().__init__()
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.lab_ext = lab_ext

        self._train_trfs = self._train_lab_trfs = self._test_trfs = CUL_datamodule.default_trfs
        self.data_ld_args = data_ld_args
        self.tst_ds = None

    def setup(self, stage=None):
        ims_path = sorted(glob(f'{self.data_dir}/**/*_im.jpg'))
        labs_path = sorted(glob(f'{self.data_dir}/**/*_msk.{self.lab_ext}'))
        val_cnt = int(self.val_ratio * len(ims_path))
        tra_cnt = len(ims_path) - val_cnt

        all_ds = CULaneDataset(ims_path, labs_path, transform=self._train_trfs,  lab_transform=self._train_lab_trfs)
        self.tra_ds, self.val_ds = random_split(all_ds, [tra_cnt, val_cnt])

    def train_dataloader(self):
        data_ld_args = self.data_ld_args if self.data_ld_args \
                            else {'batch_size':32, 'pin_memory':True}
        return DataLoader(self.tra_ds, **data_ld_args)

    def val_dataloader(self):
        data_ld_args = self.data_ld_args if self.data_ld_args \
                            else {'batch_size':32, 'pin_memory':True}
        return DataLoader(self.val_ds, **data_ld_args)

    # testing dataloader, plz exec extra-call for test_setup
    def test_setup(self, test_dir: str, lab_ext: str):
        ims_path = sorted(glob(f'{self.data_dir}/**/*.jpg'))
        labs_path = sorted(glob(f'{self.data_dir}/**/*.{lab_ext}'))
        self.tst_ds = CULaneDataset(ims_path, labs_path, transform=self._test_trfs)

    def test_dataloader(self):
        data_ld_args = self.data_ld_args if self.data_ld_args \
                            else {'batch_size':32, 'pin_memory':True}
        if self.tst_ds:
            return DataLoader(self.tst_ds, **data_ld_args)
        else:
            raise RuntimeError("test dataset is not initialized, plz call test_setup function")

    # Note : the precompute mask will be saved in place of same folder as label! 
    def precompute_mask(self):
        ims_path = sorted(glob(f'{self.data_dir}/**/*.jpg'))
        labs_path = sorted(glob(f'{self.data_dir}/**/*.{self.lab_ext}'))
        cul_ds = CULaneDataset(ims_path, labs_path)  

        for lab_path, im_path in zip(labs_path, ims_path):
            W, H = Image.open(im_path).convert('RGB').size
            lanes_cord = cul_ds._parse_cords(lab_path)
            lane_mask = cul_ds._cord2mask(lanes_cord, mask_shape=[H, W, 3])

            lane_mask = CUL_datamodule.tnsr2cvarr(lane_mask, mask=True)
            save_prefix = lab_path.split('.')[0]
            cv2.imwrite(f'{save_prefix}_msk.jpg', lane_mask)

    # utils for saving tensor image
    @staticmethod
    def tnsr2cvarr(tnsr, mask=False):
        np_arr = tnsr.cpu().detach().numpy()
        if mask:
            return np_arr

        # float32 --> uint8 with range [0, 255]
        np_arr = np.uint8( np_arr*255 / np_arr.max() )  
        # channel last format
        return cv2.cvtColor( np_arr.transpose(1,2,0), cv2.COLOR_BGR2RGB)

    # torch transformer properties : 
    @property
    def train_trfs(self):
        return self._train_trfs
    
    @train_trfs.setter
    def train_trfs(self, trfs):
        if isinstance(trfs, type(transforms)):
            raise RuntimeError(f"The given transformer should be type torch.Transformer, instead of {type(trfs)}!!")
        self._train_trfs = trfs

    @property
    def train_lab_trfs(self):
        return self._train_lab_trfs

    @train_trfs.setter
    def train_lab_trfs(self, trfs):
        if isinstance(trfs, type(transforms)):
            raise RuntimeError(f"The given transformer should be type torch.Transformer, instead of {type(trfs)}!!")
        self._train_lab_trfs = trfs

    @property
    def test_trfs(self):
        return self._test_trfs
    
    @train_trfs.setter
    def test_trfs(self, trfs):
        if isinstance(trfs, type(transforms)):
            raise RuntimeError(f"The given transformer should be type torch.Transformer, instead of {type(trfs)}!!")
        self._test_trfs = trfs

# doctest
if __name__ == '__main__':
    import doctest
    doctest.testmod()