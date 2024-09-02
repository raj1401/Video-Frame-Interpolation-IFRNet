import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from datasets import MSU_Train_Dataset, MSU_Test_Dataset
from metric import calculate_psnr, calculate_ssim
from utils import AverageMeter
import logging


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# flag if you only have one gpu, if you have multiple, single_gpu = 0
single_gpu = 1

# flag for running code
# nastaran set pretrained = 1; raj set pretrained = 0
pretrained = 1


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.epochs * args.iters_per_epoch) * math.pi))
    lr = (args.lr_start - args.lr_end) * ratio + args.lr_end
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, ddp_model):
    local_rank = args.local_rank
    print('Distributed Data Parallel Training IFRNet on Rank {}'.format(local_rank))

    if local_rank == 0:
        os.makedirs(args.log_path, exist_ok=True)
        # log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        log_path = os.path.join(args.log_path, time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))
        os.makedirs(log_path, exist_ok=True)
        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = '%(asctime)s:%(levelname)s:%(message)s'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(os.path.join(log_path, 'train.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)
        logger.info(args)

    cwd = os.getcwd()
    # path = os.path.join(cwd, 'Datasets/Vimeo90K/vimeo_triplet')
    path = os.path.join(cwd, "Datasets", "Vimeo90K", "vimeo_triplet")
    # path_msu = os.path.join(cwd, 'Datasets/MSU_Dataset/msu_triplet')
    path_msu = os.path.join(cwd, "Datasets", "MSU_Dataset", "msu_triplet")
    
    if pretrained:
        dataset_train = Vimeo90K_Train_Dataset(dataset_dir=path, augment=True)
        dataset_val = Vimeo90K_Test_Dataset(dataset_dir=path)
    else:
        dataset_train = MSU_Train_Dataset(dataset_dir=path_msu, augment=True)
        dataset_val = MSU_Test_Dataset(dataset_dir=path_msu)
    
    # Cannot use if you only have 1 GPU 
    if single_gpu:
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        sampler = DistributedSampler(dataset_train)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    
    args.iters_per_epoch = dataloader_train.__len__()
    iters = args.resume_epoch * args.iters_per_epoch
    
    # dataset_val = MSU_Test_Dataset(dataset_dir=path_msu)
    dataloader_val = DataLoader(dataset_val, batch_size=16, num_workers=8, pin_memory=True, shuffle=False, drop_last=True)

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.lr_start, weight_decay=0)

    time_stamp = time.time()
    avg_rec = AverageMeter()
    avg_geo = AverageMeter()
    avg_dis = AverageMeter()
    best_psnr = 0.0

    for epoch in range(args.resume_epoch, args.epochs):
        # Cannot use if you only have 1 GPU
        if not single_gpu:
            sampler.set_epoch(epoch)  # this is done by shuffle=True in dataloader_train
        
        for i, data in enumerate(dataloader_train):
            for l in range(len(data)):
                data[l] = data[l].to(args.device)
            img0, imgt, img1, flow, embt = data

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            lr = get_lr(args, iters)
            set_lr(optimizer, lr)

            optimizer.zero_grad()

            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt, flow)

            loss = loss_rec + loss_geo + loss_dis
            loss.backward()
            optimizer.step()

            avg_rec.update(loss_rec.cpu().data)
            avg_geo.update(loss_geo.cpu().data)
            avg_dis.update(loss_dis.cpu().data)
            train_time_interval = time.time() - time_stamp

            if (iters+1) % 100 == 0 and local_rank == 0:
                logger.info('epoch:{}/{} iter:{}/{} time:{:.2f}+{:.2f} lr:{:.5e} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e}'.format(epoch+1, args.epochs, iters+1, args.epochs * args.iters_per_epoch, data_time_interval, train_time_interval, lr, avg_rec.avg, avg_geo.avg, avg_dis.avg))
                avg_rec.reset()
                avg_geo.reset()
                avg_dis.reset()

            iters += 1
            time_stamp = time.time()

        if (epoch+1) % args.eval_interval == 0 and local_rank == 0:
            psnr = evaluate(args, ddp_model, dataloader_val, epoch, logger)
            if psnr > best_psnr:
                best_psnr = psnr
                if pretrained:
                    # torch.save(ddp_model.module.state_dict(), '{}/{}_{}_MSU.pth'.format(log_path, args.model_name, 'best'))
                    if not single_gpu:
                        torch.save(ddp_model.module.state_dict(), os.path.join(log_path, f"{args.model_name}_best_Vimeo.pth"))
                        torch.save(ddp_model.module.state_dict(), os.path.join(args.log_path, f"{args.model_name}_best_Vimeo.pth"))
                    else:
                        torch.save(ddp_model.state_dict(), os.path.join(log_path, f"{args.model_name}_best_Vimeo.pth"))
                        torch.save(ddp_model.state_dict(), os.path.join(args.log_path, f"{args.model_name}_best_Vimeo.pth"))
                else:
                    # torch.save(ddp_model.module.state_dict(), '{}/{}_{}_MSU_Trained.pth'.format(log_path, args.model_name, 'best'))
                    if not single_gpu:
                        torch.save(ddp_model.module.state_dict(), os.path.join(log_path, f"{args.model_name}_best_MSU_Trained.pth"))
                        torch.save(ddp_model.module.state_dict(), os.path.join(args.log_path, f"{args.model_name}_best_MSU_Trained.pth"))
                    else:
                        torch.save(ddp_model.state_dict(), os.path.join(log_path, f"{args.model_name}_best_MSU_Trained.pth"))
                        torch.save(ddp_model.state_dict(), os.path.join(args.log_path, f"{args.model_name}_best_MSU_Trained.pth"))
            if pretrained:
                # torch.save(ddp_model.module.state_dict(), '{}/{}_{}_MSU.pth'.format(log_path, args.model_name, 'latest'))
                if not single_gpu:
                    torch.save(ddp_model.module.state_dict(), os.path.join(log_path, f"{args.model_name}_latest_Vimeo.pth"))
                else:
                    torch.save(ddp_model.state_dict(), os.path.join(log_path, f"{args.model_name}_latest_Vimeo.pth"))
            else:
                # torch.save(ddp_model.module.state_dict(), '{}/{}_{}_MSU_Trained.pth'.format(log_path, args.model_name, 'latest'))
                if not single_gpu:
                    torch.save(ddp_model.module.state_dict(), os.path.join(log_path, f"{args.model_name}_latest_MSU_Trained.pth"))
                else:
                    torch.save(ddp_model.state_dict(), os.path.join(log_path, f"{args.model_name}_latest_MSU_Trained.pth"))
        
        # Cannot use if you only have 1 GPU
        if not single_gpu:     
            dist.barrier()


def evaluate(args, ddp_model, dataloader_val, epoch, logger):
    loss_rec_list = []
    loss_geo_list = []
    loss_dis_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(dataloader_val):
        for l in range(len(data)):
            data[l] = data[l].to(args.device)
        img0, imgt, img1, flow, embt = data

        with torch.no_grad():
            imgt_pred, loss_rec, loss_geo, loss_dis = ddp_model(img0, img1, embt, imgt, flow)

        loss_rec_list.append(loss_rec.cpu().numpy())
        loss_geo_list.append(loss_geo.cpu().numpy())
        loss_dis_list.append(loss_dis.cpu().numpy())

        for j in range(img0.shape[0]):
            psnr = calculate_psnr(imgt_pred[j].unsqueeze(0), imgt[j].unsqueeze(0)).cpu().data
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp
    
    logger.info('eval epoch:{}/{} time:{:.2f} loss_rec:{:.4e} loss_geo:{:.4e} loss_dis:{:.4e} psnr:{:.3f}'.format(epoch+1, args.epochs, eval_time_interval, np.array(loss_rec_list).mean(), np.array(loss_geo_list).mean(), np.array(loss_dis_list).mean(), np.array(psnr_list).mean()))
    return np.array(psnr_list).mean()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IFRNet')
    parser.add_argument('--model_name', default='IFRNet', type=str, help='IFRNet, IFRNet_L, IFRNet_S')
    
    if single_gpu:
        parser.add_argument('--local_rank', default=0, type=int)  # changed default: default=-1
        parser.add_argument('--world_size', default=1, type=int)
    else:
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--world_size', default=4, type=int)
    
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--eval_interval', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr_start', default=1e-4, type=float)
    parser.add_argument('--lr_end', default=1e-5, type=float)
    parser.add_argument('--log_path', default='checkpoint', type=str)
    parser.add_argument('--resume_epoch', default=0, type=int)
    parser.add_argument('--resume_path', default=None, type=str)
    args = parser.parse_args()

    # Cannot use if you only have 1 GPU
    if not single_gpu:
        dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.model_name == 'IFRNet':
        from models.IFRNet import Model
    elif args.model_name == 'IFRNet_L':
        from models.IFRNet_L import Model
    elif args.model_name == 'IFRNet_S':
        from models.IFRNet_S import Model
    elif args.model_name == 'IFRNet_S_T1':
        from models.IFRNet_S_T1 import Model
    elif args.model_name == 'IFRNet_S_T2':
        from models.IFRNet_S_T2 import Model


    # args.log_path = args.log_path + '/' + args.model_name
    args.log_path = os.path.join(args.log_path, args.model_name)
    args.num_workers = args.batch_size

    model = Model().to(args.device)
    
    if args.resume_epoch != 0:
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'))
    
    # Cannot use if you only have 1 GPU
    if single_gpu:
        train(args, model)
    else:
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        
        train(args, ddp_model)
        
        dist.destroy_process_group()
