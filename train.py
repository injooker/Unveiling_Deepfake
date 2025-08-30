import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
import random
import numpy as np
import logging
from network.models import model_selection
#from network.mesonet import Meso4, MesoInception4
from network.xception import TransferModel, Xception
#from network.mymodel import Xception_Net
from network.mymodel_bdct_dfcs_triplet_mi_loss import Xception_Net
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from sklearn.metrics import roc_auc_score
from losses.mi_loss import *
import torch.nn.functional as F
import datetime

def save_checkpoint(state, folder_path, filename='checkpoint.pth'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at {filepath}")

def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        print(f"Checkpoint loaded from {filepath}")
        return start_epoch, best_acc
    else:
        print(f"No checkpoint found at {filepath}")
        return 0, 0.0
def setup_logging(output_path):
    # 配置日志记录器
    log_file = os.path.join(output_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时打印到控制台
        ]
    )
    return logging.getLogger()
def set_random_seed(seed=1234):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)  # 设置 Python 内置的 random 模块种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed(seed)  # 设置当前设备的 CUDA 随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有设备的 CUDA 随机种子
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动调优，以确保一致性
    print("设置随机种子： ",seed)
def main():
    args = parse.parse_args()
    # 设置随机种子
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join('./output', timestamp)
    set_random_seed(seed=1234)
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    lr = args.lr
   
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # 设置日志
    logger = setup_logging(output_path)
    torch.backends.cudnn.benchmark=True
    print('train_dataset: ', train_list)
    print('val_dataset: ', val_list)
    print('lr: ',lr)
    train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'],train = True)
    val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda:0")
    model = Xception_Net()
    model = model.to(device)
    loss_function = loss_functions(method='mi',
                                            mi_calculator="kl", temperature=1.5,
                                            bml_method='mi', scales=[1,2,10],
                                            dec_loss=True,
                                            gia_loss=True,
                                            device='cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    best_model_wts = model.state_dict()
    best_auc = 0.0
    iteration = 0
    loss_history = []
    if args.continue_train and os.path.exists(args.model_path):
        start_epoch, best_auc, loss_history = load_checkpoint(args.model_path, model, optimizer)

    for epoch in range(epoches):
        logger.info('Epoch {}/{}'.format(epoch+1, epoches))
        logger.info('-'*10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        avg_loss = []
        avg_ce_loss=[]
        avg_global_mi_loss=[]
        avg_local_loss=[]
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            
            outputs = model(image)
            _, preds = torch.max(outputs['out'].data, 1)
            # loss = criterion(outputs, labels)
            losses = loss_function.criterion(outputs,labels)
            loss = loss_function.balance_mult_loss(losses)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 20):
                logger.info('iteration {} train loss: {:.6f} Acc: {:.6f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        logger.info('epoch train loss: {:.6f} Acc: {:.6f}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            batch_count = 0  # Initialize a counter for batches
            all_labels = []
            all_probs = []
            val_loss = 0.0
            val_corrects = 0.0
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                
              
                outputs = model(image)  # 这里是模型的前向传播        
                probs = F.softmax(outputs['out'], dim=1)[:, 1]  # 假设类别 1 是正类
                _, preds = torch.max(outputs['out'].data, 1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                losses = loss_function.criterion(outputs,labels)
                loss = loss_function.balance_mult_loss(losses)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
                batch_count += 1
                if batch_count % 20 == 0:  # Print every 20 batches
                    batch_acc = torch.sum(preds == labels.data).to(torch.float32) / image.size(0)
                    logger.info('Batch {}/{} - Acc: {:.6f}'.format(batch_count, len(val_loader), batch_acc))
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            epoch_auc = roc_auc_score(all_labels, all_probs)
            loss_history.append({'epoch': epoch+1,'val_loss': epoch_loss, 'auc': epoch_auc})
            logger.info('epoch val loss: {:.6f} Acc: {:.6f} AUC: {:.6f}'.format(epoch_loss, epoch_acc,epoch_auc))
            if epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = model.state_dict()
        scheduler.step()
        
        # 保存每个 epoch 的模型参数为 .pkl 文件
        model_filename = os.path.join(output_path, f'epoch_{epoch + 1}_best_model.pkl')
        torch.save(model.module.state_dict(), model_filename)
        logger.info(f'Saved model parameters for epoch {epoch + 1} at {model_filename}')
         # 保存检查点
        # checkpoint = {
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'best_auc': best_auc,
        #     'loss_history': loss_history,
        # }
        # save_checkpoint(checkpoint, output_path, f'checkpoint_epoch_{epoch+1}.pth')
    logger.info('Best val Auc: {:.6f}'.format(best_auc))
    model.load_state_dict(best_model_wts,strict = False)
    torch.save(model.module.state_dict(), os.path.join(output_path, "cdf2_miloss_best.pkl"))




if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='xception_benchmark')
    parse.add_argument('--train_list', '-tl' , type=str, default = '/data1/sqh_data/sqh/Deepfake-Detection/Data/DeepfakeBench/FF++/train_balance.txt')
    parse.add_argument('--val_list', '-vl' , type=str, default = '/data1/sqh_data/sqh/Deepfake-Detection/Data/DeepfakeBench/FF++/val.txt')
    #Deepfakes NeuralTextures FaceSwap Face2Face
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default='15')
    parse.add_argument('--lr', '-lr', type=float, default=5e-4)
    parse.add_argument('--model_name', '-mn', type=str, default='xception_dfcs_cdf2_miloss_benchmark.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/df_xception_c0_299/1_df_c0_299.pkl')
    main()
