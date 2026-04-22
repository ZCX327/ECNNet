import os
import json
import sys
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from utils.dataIo2 import ImageMaskDataset
from utils.utils_z import compute_metrics,compute_metrics_m
from utils.loss import BCEDiceBoundaryLoss, TverskyLoss_Binary
from unet import UNet, UNetPP, UNet3Plus
from RexNeXtUnet import ResNeXtUNet, ResNeXtUNetW, ConvNeXtUNet, ConvNeXtUNet2
from torch.optim import AdamW
from pytorch_optimizer import Lookahead
from test_model import MultiDatasetEvaluator
from lib.PraNet_Res2Net import PraNet
from Seg_UKAN.archs import UKAN
from PolypPVT.pvt import PolypPVT

# 定义外部测试数据集
external_datasets = {
    "Kvasir": ("./data/main/TestDataset/Kvasir/images", "./data/main/TestDataset/Kvasir/masks"),
    "ETIS": ("./data/main/TestDataset/ETIS-LaribPolypDB/images", "./data/main/TestDataset/ETIS-LaribPolypDB/masks"),
    "ColonDB": ("./data/main/TestDataset/CVC-ColonDB/images", "./data/main/TestDataset/CVC-ColonDB/masks"),
    "ClinicDB": ("./data/main/TestDataset/CVC-ClinicDB/images", "./data/main/TestDataset/CVC-ClinicDB/masks"),
    "EndoScene": ("./data/main/TestDataset/CVC-300/images", "./data/main/TestDataset/CVC-300/masks")
}


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_model(config):
    net_name = config["experiment"]["network_name"]
    if net_name == "ConvNeXtUNet":
        return ConvNeXtUNet(num_classes=1)
    elif net_name == "ConvNeXtUNet2":
        return ConvNeXtUNet2(num_classes=1)
    elif net_name == "UNet":
        return UNet(n_channels=3, n_classes=1, bilinear=False)
    elif net_name == "UNetPP":
        return UNetPP()  # 注意：你原代码中 UNetPP 是从 unet 导入的
    elif net_name == "PraNet":
        return PraNet()  
    elif net_name == "ukan":
        return UKAN(num_classes=1)  
    elif net_name == "PolypPVT":
        return PolypPVT()  
    else:
        raise ValueError(f"Unknown network: {net_name}")


def get_criterion(config):
    loss_type = config["loss"]["type"]
    if loss_type == "BCEDiceBoundaryLoss":
        return BCEDiceBoundaryLoss(
            bce_weight=config["loss"]["bce_weight"],
            dice_weight=config["loss"]["dice_weight"],
            boundary_w0=config["loss"]["boundary_w0"],
            boundary_sigma=config["loss"]["boundary_sigma"]
        )
    elif loss_type == "TverskyLoss_Binary":
        return TverskyLoss_Binary()
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")


def get_dataloaders(config):
    dataset_name = config["experiment"]["dataset_name"]
    data_root = Path(config["paths"]["data_root"])
    h, w = config["training"]["img_height"], config["training"]["img_width"]

    if dataset_name == "ClinicDB":
        img_path = data_root / "CVC-ClinicDB/Original"
        mask_path = data_root / "CVC-ClinicDB/Ground Truth"
    elif dataset_name == "kvasir":
        img_path = data_root / "kvasir-seg/images"
        mask_path = data_root / "kvasir-seg/masks"
    elif dataset_name == "main":
        img_path = data_root / "main/TrainDataset/image"
        mask_path = data_root / "main/TrainDataset/mask"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = ImageMaskDataset(img_path, mask_path, img_size=(h, w))

    val_pct = config["training"]["val_percent"]
    test_pct = config["training"]["test_percent"]
    total = len(dataset)
    
    n_val = int(total * val_pct)
    n_test = int(total * test_pct)
    
    # 如果比例 > 0 但计算结果为 0，则设为 1（可选）
    if val_pct > 0 and n_val == 0:
        n_val = 1
    if test_pct > 0 and n_test == 0:
        n_test = 1
    
    n_train = total - n_val - n_test
    
    if n_train <= 0:
        raise ValueError(f"训练集大小必须 > 0，当前总样本 {total}，验证 {n_val}，测试 {n_test}")
    
    # 动态构建分割列表（只包含 >0 的部分）
    split_sizes = [n_train]
    split_names = ['train']
    
    if n_val > 0:
        split_sizes.append(n_val)
        split_names.append('val')
    if n_test > 0:
        split_sizes.append(n_test)
        split_names.append('test')
    
    # 执行分割
    splits = random_split(
        dataset,
        split_sizes,
        generator=torch.Generator().manual_seed(config["training"]["random_seed"])
    )
    
    # 映射回变量
    loader_dict = {}
    for name, subset in zip(split_names, splits):
        batch_size = config["training"]["batch_size"]
        shuffle = (name == 'train')
        loader_dict[name] = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = loader_dict['train']
    val_loader = loader_dict.get('val', None)
    test_loader = loader_dict.get('test', None)

    return train_loader, val_loader, test_loader


def save_config(config, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def main(config_path="config.json"):
    evaluator = MultiDatasetEvaluator(
        dataset_dict=external_datasets,
        img_size=(224, 224),
        batch_size=32,
        device="cuda"
    )
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set random seed
    random.seed(config["training"]["random_seed"])
    torch.manual_seed(config["training"]["random_seed"])
    torch.cuda.manual_seed_all(config["training"]["random_seed"])

    # Create output dir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['experiment']['user_defined_name']}_{timestamp}_{config['experiment']['network_name']}_{config['experiment']['dataset_name']}"
    output_dir = Path(config["paths"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to log
    sys.stdout = Logger(output_dir / "train.log")

    # Save config copy
    save_config(config, output_dir / "config.json")

    # Print config
    print("Loaded configuration:")
    print(json.dumps(config, indent=2))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model, loss, optimizer
    model = get_model(config).to(device)
    # model.load_state_dict(torch.load("./outputs/output_20251218_203754_ukan_main/epoch_end.pth"))
    criterion = get_criterion(config)

    base_opt = AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"]
    )
    optimizer = Lookahead(
        base_opt,
        k=config["optimizer"]["lookahead_k"],
        alpha=config["optimizer"]["lookahead_alpha"]
    )

    # Paths
    csv_file = output_dir / "training_metrics.csv"
    best_model_path_val = output_dir / "best_model_V.pth"
    best_model_path_test = output_dir / "best_model_T.pth"
    best_model_path_colondb_dice = output_dir / "best_model_c.pth"
    end_model_path = output_dir / "epoch_end.pth"

    # Metrics DataFrame
    metrics_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'val_loss', 'test_loss',
        'val_dice', 'val_iou', 'test_dice', 'test_iou'
    ])

    # Training loop
    best_val_dice = 0.0
    best_test_dice = 0.0
    best_t = 0.0
    val_loss = val_dice = val_iou = 0
    test_loss = test_dice = test_iou = 0
    

    for epoch in range(config["training"]["epochs"]):
        print(f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
        print('-' * 10)

        # Train
        model.train()
        epoch_loss = 0
        num_batches_used = 0
        max_batches = 80  
        max_batches = min(max_batches,len(train_loader))
        for batch in tqdm(train_loader, desc='Training',total=max_batches):
            if num_batches_used >= max_batches:
                break
            inputs, masks = batch["img_tensor"].to(device), batch["mask_tensor"].to(device)
            optimizer.zero_grad()
            outputs,outputs2 = model(inputs)
            
            
            loss, _ = criterion(outputs, masks)
            loss2, _ = criterion(outputs2, masks)
            loss = loss+loss2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches_used += 1
        train_loss = epoch_loss / max_batches
        print(f'train_loss: {train_loss:.6f}')

        # Validate
        if val_loader is not None:
            model.eval()
            val_loss = val_dice = val_iou = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    inputs, masks = batch["img_tensor"].to(device), batch["mask_tensor"].to(device)
                    outputs,_ = model(inputs)
                    loss, _ = criterion(outputs, masks)
                    val_loss += loss.item()
                    dice, iou = compute_metrics(outputs, masks)
                    # print(f"dice:{dice},iou:{iou},mdice:{mdice},miou:{miou}")
                    val_dice += dice
                    val_iou += iou
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            val_iou /= len(val_loader)

        # Test
        if test_loader is not None:
            model.eval()
            test_loss = test_dice = test_iou = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Test'):
                    inputs, masks = batch["img_tensor"].to(device), batch["mask_tensor"].to(device)
                    outputs,_ = model(inputs)
                    loss, _ = criterion(outputs, masks)
                    test_loss += loss.item()
                    dice, iou = compute_metrics(outputs, masks)
                    test_dice += dice
                    test_iou += iou
                   
            test_loss /= len(test_loader)
            test_dice /= len(test_loader)
            test_iou /= len(test_loader)

        # Log
        print(f'val_loss: {val_loss:.6f}, val_dice: {val_dice:.4f}, val_iou: {val_iou:.4f}')
        print(f'test_loss: {test_loss:.6f}, test_dice: {test_dice:.4f}, test_iou: {test_iou:.4f}')
        print('-' * 40)

        # 测试完成后，自动跑外部测试集
        if epoch % 1000 == 1000:
            external_csv = output_dir / f"ExternalDataset_epoch{epoch+1}.csv"
            df = evaluator.evaluate(
                model=model,
                criterion=criterion,
                save_csv=external_csv
            )
            colondb_dice = df[df['dataset'] == 'ColonDB']['dice'].values[0]
            print(f"ColonDB dice 值: {colondb_dice}")
            if colondb_dice >= best_t:
                best_t = colondb_dice
                torch.save(model.state_dict(), best_model_path_colondb_dice)
                print(f"✅ New best model saved (Dice={colondb_dice:.4f})")
                

        # Save metrics
        new_row = pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'test_dice': test_dice,
            'test_iou': test_iou
        }])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        metrics_df.to_csv(csv_file, index=False)

        # Save best models
        if val_dice > best_val_dice and val_loader is not None:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path_val)
            print(f"✅ New best validation model saved (Dice={val_dice:.4f})")

        if test_dice > best_test_dice and test_loader is not None:
            best_test_dice = test_dice
            torch.save(model.state_dict(), best_model_path_test)
            print(f"✅ New best test model saved (Dice={test_dice:.4f})")
            
        if epoch == config["training"]["epochs"]-1:
            torch.save(model.state_dict(), end_model_path)
            print(f"✅ Model saved!!!")
 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to config JSON file")
    args = parser.parse_args()
    main(args.config)