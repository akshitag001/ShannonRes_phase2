import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.models.nafnet import NAFNet

class TestDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npy')]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.input_dir, filename)
        img = np.load(path).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0), filename

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fast initialization
    model = NAFNet(
        in_channels=1, out_channels=1, width=32,
        enc_blk_nums=[1, 1, 1, 14], middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 14], scale=2, use_film=True
    ).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'model_nafnet_film_v2.pth')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    dataset = TestDataset(input_dir)
    # Using multiple workers speeds up I/O significantly
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    count = 0
    failures = 0
    
    with torch.no_grad():
        for batch_imgs, batch_filenames in loader:
            batch_imgs = batch_imgs.to(device)
            try:
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    preds = model(batch_imgs)
                
                preds = preds.to(torch.float32)
                
                # Sanitize outputs for strictly valid metrics
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
                preds = torch.clamp(preds, 0.0, 1.0)
                
                preds_np = preds.squeeze(1).cpu().numpy().astype(np.float32)
                
                for i in range(len(batch_filenames)):
                    out_path = os.path.join(output_dir, batch_filenames[i])
                    np.save(out_path, preds_np[i])
                    count += 1
            except Exception as e:
                print(f"Error processing batch containing {batch_filenames[0]}: {e}")
                failures += len(batch_filenames)
                
    print(f"Inference complete. Successfully processed: {count}. Failures: {failures}.")

if __name__ == '__main__':
    main()
