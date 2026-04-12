def set_template(args):
    if args.template == 'KernelPredict':
        args.task = "PretrainKernel"
        args.model = "Kernel"
        args.save = "Kernel_MANet_DBVSR_64_128_Pretrain"      
        args.data_train = 'REDS_ONLINE'  
        # args.data_train = 'REDS_HRLR'  
        args.dir_data = '../dataset/jilin189_BlurDown_Gaussian/'
        args.data_test = 'REDS_HRLR'  
        args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_002009010/'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5   
        args.n_frames_per_video = 100 
        args.est_ksize = 13
        args.loss = '1*L1'
        args.lr = 1e-4
        args.lr_decay = 20     
        args.save_middle_models = True
        args.save_images = True
        args.epochs = 30
        args.batch_size = 16
        args.resume = True
        args.load = args.save

    elif args.template == 'VideoSR':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Ablation_1_large"
        # args.data_train = 'REDS_ONLINE'
        args.data_train = 'REDS_HRLR'
        args.dir_data = '../dataset/jilin189_BlurDown_Gaussian/'
        args.data_test = 'REDS_HRLR'
        # args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_002009010/'
        args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_001/'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 128           
        args.n_cond = 128           
        args.est_ksize = 13
        args.extra_RBS = 3
        args.recons_RBS = 20
        args.loss = '1*L1'
        args.lr = 1e-4
        # args.lr = 8e-5
        args.lr_decay = 50
        args.save_middle_models = True
        args.save_images = False
        args.epochs = 100
        args.batch_size = 8

        # args.resume = True
        # args.load = args.save

    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))

