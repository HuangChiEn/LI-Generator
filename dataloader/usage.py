from ULane_loader import CUL_datamodule

if __name__ == "__main__":
    import cv2
    from easy_configer.Configer import Configer

    cfger = Configer()
    cfger.cfg_from_str(
    '''
    data_dir = ../data/driver_23_30frame@str
    test_dir = ../data/driver_161_90frame@str
    # may be we will have pre-computed segmap, then it would be jpg
    lab_ext = txt@str

    [dataloader]
    batch_size = 32@int
    num_workers = 4@int
    pin_memory = True@bool
    shuffle = True@bool
    '''
    )

    #============ Training usage with supporting val_split ====================
    # test train dataloaders with val split
    cul_dm = CUL_datamodule(cfger.data_dir, cfger.lab_ext, data_ld_args=cfger.dataloader)
    cul_dm.setup()

    # get utils from CUL_datamodule
    tnsr2cvarr = CUL_datamodule.tnsr2cvarr
    for idx, (im, lane_msk) in enumerate(cul_dm.train_dataloader()):
        cv2.imwrite(f'./test/train_{idx}.jpg', tnsr2cvarr(im[0]) )
        cv2.imwrite(f'./test/train_msk_{idx}.jpg', tnsr2cvarr(lane_msk[0], mask=True) )
        if idx == 1 : break

    #========================= Testing usage ===========================
    cul_dm.test_setup(cfger.test_dir, cfger.lab_ext)

    for idx, (im, lane_msk) in enumerate(cul_dm.test_dataloader()):
        cv2.imwrite(f'./test/test_{idx}.jpg', tnsr2cvarr(im[0]) )
        cv2.imwrite(f'./test/test_msk_{idx}.jpg', tnsr2cvarr(lane_msk[0], mask=True) )
        if idx == 1 : break

    #=================== Mask generation before training =======================
    # Note : the precompute mask will be saved in place of same folder as label! 
    # recommend runtime for little overhead..maybe..
    #cul_dm.precompute_mask()