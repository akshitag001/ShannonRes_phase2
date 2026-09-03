import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.model import RestorationCNN
from src.models.nafnet import NAFNet

class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.input_dir, filename)
        
        # Load and prepare shape (1, H, W)
        arr = np.load(filepath).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, filename

def main():
    parser = argparse.ArgumentParser(description="Inference for Image Restoration")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output .npy files")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[os.path.join("weights", "best_model_seed2.pth")], help="Paths to model checkpoints for ensembling")
    parser.add_argument("--model", type=str, default="restoration_cnn", help="Model architecture")
    parser.add_argument("--fast", action="store_true", help="Skip TTA for faster inference")
    parser.add_argument("--with_uncertainty", action="store_true", help="Output uncertainty maps alongside restored images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}")
    
    # Load all models for ensembling
    models = []
    
    for ckpt_path in args.checkpoints:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please train the model first.")
            
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        scale = checkpoint['scale']
        
        if args.model == "nafnet":
            model = NAFNet(
                in_channels=config.get('in_channels', 1),
                out_channels=config.get('out_channels', 1),
                width=config.get('width', 32),
                enc_blk_nums=config.get('enc_blk_nums', [1, 1, 1, 28]),
                middle_blk_num=config.get('middle_blk_num', 1),
                dec_blk_nums=config.get('dec_blk_nums', [1, 1, 1, 1]),
                scale=scale
            ).to(device)
        else:
            model = RestorationCNN(
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
                num_features=config['num_features'],
                num_res_blocks=config['num_res_blocks'],
                scale=scale
            ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=not args.with_uncertainty)
        if args.with_uncertainty:
            model.predict_uncertainty = True
        model.eval()
        
        try:
            model = torch.compile(model)
            print(f"Successfully compiled model from {ckpt_path} with torch.compile()")
        except Exception as e:
            print(f"Warning: torch.compile() failed, falling back to uncompiled model. Error: {e}")
            
        models.append(model)
        
    print(f"Successfully loaded {len(models)} model(s) for ensembling.")
    print(f"Inference Mode: {'Fast (No TTA)' if args.fast else 'High Quality (8x TTA)'}")

    # DataLoader for batching
    dataset = InferenceDataset(args.input_dir)
    # Use batch_size from config or default to 32 for inference
    batch_size = config.get('batch_size', 32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def inference_with_tta(x, model):
        outputs = []
        sigmas = []
        for k in range(4):
            # Rotate
            rot_x = torch.rot90(x, k, dims=[2, 3])
            if args.with_uncertainty:
                out_rot, sigma_rot = model(rot_x)
                sigmas.append(torch.rot90(sigma_rot, -k, dims=[2, 3]))
            else:
                out_rot = model(rot_x)
            outputs.append(torch.rot90(out_rot, -k, dims=[2, 3]))
            
            # Rotate + Flip
            flip_x = torch.flip(rot_x, [3])
            if args.with_uncertainty:
                out_flip, sigma_flip = model(flip_x)
                sigmas.append(torch.rot90(torch.flip(sigma_flip, [3]), -k, dims=[2, 3]))
            else:
                out_flip = model(flip_x)
            outputs.append(torch.rot90(torch.flip(out_flip, [3]), -k, dims=[2, 3]))
            
        if args.with_uncertainty:
            return torch.mean(torch.stack(outputs), dim=0), torch.mean(torch.stack(sigmas), dim=0)
        return torch.mean(torch.stack(outputs), dim=0)

    with torch.no_grad():
        for batch_tensors, filenames in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            # Inference with optional 8x TTA across all ensembled models
            ensemble_outputs = []
            ensemble_sigmas = []
            for m in models:
                if args.fast:
                    if args.with_uncertainty:
                        out_t, sig_t = m(batch_tensors)
                        ensemble_outputs.append(out_t)
                        ensemble_sigmas.append(sig_t)
                    else:
                        ensemble_outputs.append(m(batch_tensors))
                else:
                    if args.with_uncertainty:
                        out_t, sig_t = inference_with_tta(batch_tensors, m)
                        ensemble_outputs.append(out_t)
                        ensemble_sigmas.append(sig_t)
                    else:
                        ensemble_outputs.append(inference_with_tta(batch_tensors, m))
                
            outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)
            outputs = outputs.cpu().numpy()
            
            if args.with_uncertainty:
                sigmas = torch.mean(torch.stack(ensemble_sigmas), dim=0)
                sigmas = sigmas.cpu().numpy()
                unc_dir = os.path.join(args.output_dir, "uncertainty_maps")
                os.makedirs(unc_dir, exist_ok=True)
            
            for i in range(len(filenames)):
                filename = filenames[i]
                # Squeeze channel dim (1, H, W) -> (H, W) to match input shape
                out_arr = outputs[i].squeeze(0)
                
                save_path = os.path.join(args.output_dir, filename)
                np.save(save_path, out_arr)
                
                if args.with_uncertainty:
                    sigma_arr = sigmas[i].squeeze(0)
                    np.save(os.path.join(unc_dir, filename), sigma_arr)
                    
                    # Normalize and save as heatmap PNG
                    sigma_norm = (sigma_arr - sigma_arr.min()) / (sigma_arr.max() - sigma_arr.min() + 1e-8)
                    png_filename = filename.replace('.npy', '.png')
                    plt.imsave(os.path.join(unc_dir, png_filename), sigma_norm, cmap='inferno')
                
    print(f"Successfully processed {len(dataset)} images and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
