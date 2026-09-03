import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.cuda.amp import autocast, GradScaler
import time
import csv
import lpips
import random
import numpy as np
from tqdm import tqdm

from src.dataset import ImageRestorationDataset
from src.models.nafnet import NAFNet
from src.losses import RestorationCombinedLoss
from src.metrics import MetricsEvaluator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/final_model.yaml')
    parser.add_argument('--resume', type=str, default='weights/model_nafnet_v2.pth', help='Path to checkpoint to warm-start from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    # Train = Phase 1 + non-validation Phase 2
    ds_phase1 = ImageRestorationDataset(config['train_gt_dir'], config['train_noisy_dir'], gt_crop_size=config['gt_crop_size'], synthesize_prob=config.get('synthesize_prob', 0.5))
    ds_phase2_train = ImageRestorationDataset(config['phase2_gt_dir'], config['phase2_noisy_dir'], gt_crop_size=config['gt_crop_size'], synthesize_prob=0.0, is_val=False)
    
    # Validation uses Phase 2 entirely for fair benchmark (478 images)
    ds_phase2_val = ImageRestorationDataset(config['phase2_gt_dir'], config['phase2_noisy_dir'], gt_crop_size=config['gt_crop_size'], synthesize_prob=0.0, is_val=True)
    
    idx = list(range(len(ds_phase2_val)))
    random_state = random.getstate()
    random.seed(42)
    random.shuffle(idx)
    random.setstate(random_state)
    
    val_size = int(config.get('val_split_ratio', 0.1) * len(ds_phase2_val))
    val_idx = idx[-val_size:]
    val_subset = Subset(ds_phase2_val, val_idx)
    
    train_phase2_subset = Subset(ds_phase2_train, idx[:-val_size])
    train_full_ds = ConcatDataset([ds_phase1, train_phase2_subset])

    train_loader = DataLoader(train_full_ds, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    # FiLM-Conditioned NAFNet
    model = NAFNet(
        in_channels=1, out_channels=1, width=32,
        enc_blk_nums=[1, 1, 1, 14], middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 14], scale=2, use_film=True
    ).to(device)

    # Note: FiLM layer weights are automatically zero-initialized in the architecture definition (film.py / nafnet.py)
    # to preserve the warm-started baseline behavior at step 0.
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    criterion = RestorationCombinedLoss(
        lpips_model=lpips_model,
        lambda_charbonnier=config.get('lambda_charbonnier', 1.0),
        lambda_ssim=config.get('lambda_ssim', 0.5),
        lambda_sobel=config.get('lambda_sobel', 0.1),
        lambda_lpips=config.get('lambda_lpips', 0.3),
        device=device
    )

    evaluator = MetricsEvaluator(lpips_model=lpips_model, device=device)

    start_epoch = 0
    best_psnr = 0.0
    patience_counter = 0
    patience_limit = config.get('patience', 15)

    if args.resume and os.path.isfile(args.resume):
        print(f"Warm-starting from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # strict=False allows the newly initialized FiLM parameters to be added safely to the pre-trained backbone
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("Warning: Expected warm-start checkpoint not found. Training from scratch.")

    os.makedirs(config['save_dir'], exist_ok=True)
    metrics_log_file = os.path.join(config['save_dir'], 'train_log.csv')
    
    with open(metrics_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_lpips', 'seconds'])

    scaler = GradScaler()

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        # Dataset returns noisy, gt, cond1, cond2, but NAFNet extracts the embedding directly from noisy
        for noisy, gt, _, _ in pbar:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred = model(noisy)
                loss, charb, ssim, sobel, lpips_val = criterion(pred, gt)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'SSIM_Loss': f"{ssim:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_psnr, val_ssim, val_lpips = 0.0, 0.0, 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
            for noisy, gt, _, _ in pbar_val:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                psnr, ssim, lpips_val = evaluator.evaluate(pred, gt)
                val_psnr += psnr
                val_ssim += ssim
                val_lpips += lpips_val
                
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        val_lpips /= len(val_loader)
        
        epoch_duration = time.time() - epoch_start_time
        
        with open(metrics_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, val_psnr, val_ssim, val_lpips, epoch_duration])
            
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f}")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            save_path = os.path.join(config['save_dir'], 'model_nafnet_film_v2.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'config': config
            }, save_path)
            print(f"Saved new best model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    main()
