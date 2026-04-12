import torch
import data
import model
import loss
import option
from trainer.trainer_kernel import Trainer_Kernel
from trainer.trainer_flow_video import Trainer_Flow_Video
from logger import logger
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# import multiprocessing


args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)

print("Selected task: {}".format(args.task))
model = model.Model(args, chkp)    # 初始化模型，训练过程中还要把数据导到model里
loss = loss.Loss(args, chkp) if not args.test_only else None
loader = data.Data(args)

if args.task == 'PretrainKernel':
    t = Trainer_Kernel(args, loader, model, loss, chkp)        # 相当于初始化参数
elif args.task == 'FlowVideoSR':
    t = Trainer_Flow_Video(args, loader, model, loss, chkp)
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))


while not t.terminate():
    # multiprocessing.set_start_method('spawn')
    t.train()   # 真正训练
    t.test()

chkp.done()
