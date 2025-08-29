import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
from network.xception import TransferModel, Xception
from network.mymodel_bdct_dfcs_triplet_mi_loss import Xception_Net
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score  # 导入 F1 score 计算函数
from dataset.mydataset import MyDataset
import torch.nn.functional as F
import logging
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
def main():
    args = parse.parse_args()
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path
    name = "testlog"
    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    # 设置日志
    logger = setup_logging(output_path)
    torch.backends.cudnn.benchmark=True
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    test_dataset_size = len(test_dataset)
    corrects = 0
    acc = 0
    #model = torchvision.models.densenet121(num_classes=2)
    model = Xception_Net()
   
    model.load_state_dict(torch.load(model_path))
   
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # 使用多GPU
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # 指定使用GPU 0, 1, 2, 3
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        batch_count = 0 # Initialize a counter for batches
        all_labels = []
        all_probs = []
        all_preds = []
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            probs = F.softmax(outputs['out'], dim=1)[:, 1]  # 假设类别 1 是正类
            _, preds = torch.max(outputs['out'].data, 1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())  # 存储预测结果
            corrects += torch.sum(preds == labels.data).to(torch.float32)
            batch_count += 1
            if batch_count % 20 == 0:  # Print every 20 batches
                batch_acc = torch.sum(preds == labels.data).to(torch.float32) / image.size(0)
                logger.info('Batch {}/{} - Acc: {:.6f}'.format(batch_count, len(test_loader), batch_acc))   
        epoch_acc = corrects / test_dataset_size
        epoch_auc = roc_auc_score(all_labels, all_probs)
        epoch_f1 = f1_score(all_labels, all_preds)  # 计算 F1 score

        # 记录 epoch 的评估结果
        logger.info('Epoch Acc: {:.6f} AUC: {:.6f} F1: {:.6f}'.format(epoch_acc, epoch_auc, epoch_f1))
    



if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--test_list', '-tl', type=str, default='/data1/sqh_data/sqh/Deepfake-Detection/Data/DeepfakeBench/FF++/val.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='/data1/sqh_data/sqh/Deepfake-Detection/output/20250131_101826/epoch_21_best_model.pkl')
    main()