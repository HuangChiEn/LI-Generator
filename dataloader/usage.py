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
    batch_size = 32@int
    num_work = 12@int
    '''
    )

    #============ Training usage with supporting val_split ====================
    # test train dataloaders with val split
    cul_dm = CUL_datamodule(cfger.data_dir, cfger.lab_ext)
    cul_dm.setup()

    for idx, (im, lane_msk) in enumerate(cul_dm.train_dataloader()):
        cv2.imwrite(f'./train_{idx}.jpg', im[0])
        cv2.imwrite(f'./train_msk_{idx}.jpg', lane_msk[0])
        if idx == 1 : break

    #========================= Testing usage ===========================
    cul_dm.test_setup(cfger.test_dir, cfger.lab_ext)

    for idx, (im, lane_msk) in enumerate(cul_dm.test_dataloader()):
        cv2.imwrite(f'./test_{idx}.jpg', im[0])
        cv2.imwrite(f'./test_msk_{idx}.jpg', lane_msk[0])
        if idx == 1 : break

    #=================== Mask generation before training =======================
    # Note : the precompute mask will be saved in place of same folder as label! 
    # recommend runtime for little overhead..maybe..
    #cul_dm.precompute_mask()