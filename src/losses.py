import torch
import torch.nn as nn
import lpips
from torchmetrics.functional.image import structural_similarity_index_measure

class RestorationLoss(nn.Module):
    def __init__(self, lpips_model, lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.5, device='cuda'):
        super(RestorationLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        
        self.l1_loss = nn.L1Loss()
        # LPIPS expects 3 channels, we will repeat grayscale channels
        self.lpips_loss = lpips_model
        self.device = device

    def forward(self, pred, gt):
        # 1. L1 Loss
        loss_pixel = self.l1_loss(pred, gt)
        pixel_weight = self.lambda_l1
        l1_val = loss_pixel.item()
        
        # 2. SSIM Loss (1 - SSIM)
        # ssim outputs a tensor scalar
        ssim_val = structural_similarity_index_measure(pred, gt, data_range=1.0)
        loss_ssim = 1.0 - ssim_val
        
        # 3. LPIPS Loss
        # LPIPS expects inputs in [-1, 1] and 3 channels
        pred_3c = pred.repeat(1, 3, 1, 1)
        gt_3c = gt.repeat(1, 3, 1, 1)
        
        # Convert [0, 1] to [-1, 1]
        pred_lpips_in = pred_3c * 2.0 - 1.0
        gt_lpips_in = gt_3c * 2.0 - 1.0
        
        loss_lpips = self.lpips_loss(pred_lpips_in, gt_lpips_in).mean()
        
        # Total Loss
        total_loss = (pixel_weight * loss_pixel + 
                      self.lambda_ssim * loss_ssim + 
                      self.lambda_lpips * loss_lpips)
                      
        return total_loss, l1_val, ssim_val.item(), loss_lpips.item()

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps) - self.eps)
        return loss

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        # Sobel filters for X and Y directions
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.l1_loss = nn.L1Loss()
        
    def _get_gradients(self, img):
        # We assume img is [B, 1, H, W]
        # Pad first to preserve spatial dimensions
        img_pad = nn.functional.pad(img, (1, 1, 1, 1), mode='reflect')
        grad_x = nn.functional.conv2d(img_pad, self.sobel_x.to(device=img.device, dtype=img.dtype))
        grad_y = nn.functional.conv2d(img_pad, self.sobel_y.to(device=img.device, dtype=img.dtype))
        return grad_x, grad_y
        
    def forward(self, pred, gt):
        pred_grad_x, pred_grad_y = self._get_gradients(pred)
        gt_grad_x, gt_grad_y = self._get_gradients(gt)
        
        loss_x = self.l1_loss(pred_grad_x, gt_grad_x)
        loss_y = self.l1_loss(pred_grad_y, gt_grad_y)
        return loss_x + loss_y

class RestorationCharbonnierSobelLoss(nn.Module):
    def __init__(self, lpips_model, lambda_charbonnier=1.0, lambda_sobel=0.1, lambda_lpips=0.05, device='cuda'):
        super(RestorationCharbonnierSobelLoss, self).__init__()
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_sobel = lambda_sobel
        self.lambda_lpips = lambda_lpips
        
        self.charbonnier_loss = CharbonnierLoss()
        self.sobel_loss = SobelEdgeLoss()
        self.lpips_loss = lpips_model
        self.device = device

    def forward(self, pred, gt):
        # 1. Charbonnier Loss
        loss_charb = self.charbonnier_loss(pred, gt)
        
        # 2. Sobel Edge Loss
        loss_sobel = self.sobel_loss(pred, gt)
        
        # 3. LPIPS Loss
        pred_3c = pred.repeat(1, 3, 1, 1)
        gt_3c = gt.repeat(1, 3, 1, 1)
        pred_lpips_in = pred_3c * 2.0 - 1.0
        gt_lpips_in = gt_3c * 2.0 - 1.0
        
        loss_lpips = self.lpips_loss(pred_lpips_in, gt_lpips_in).mean()
        
        # Total Loss
        total_loss = (self.lambda_charbonnier * loss_charb + 
                      self.lambda_sobel * loss_sobel + 
                      self.lambda_lpips * loss_lpips)
                      
        return total_loss, loss_charb.item(), loss_sobel.item(), loss_lpips.item()

class RestorationCombinedLoss(nn.Module):
    def __init__(self, lpips_model, lambda_charbonnier=1.0, lambda_ssim=0.5, lambda_sobel=0.1, lambda_lpips=0.3, device='cuda'):
        super(RestorationCombinedLoss, self).__init__()
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_ssim = lambda_ssim
        self.lambda_sobel = lambda_sobel
        self.lambda_lpips = lambda_lpips
        
        self.charbonnier_loss = CharbonnierLoss(eps=1e-6)
        self.sobel_loss = SobelEdgeLoss()
        self.lpips_loss = lpips_model
        self.device = device

    def forward(self, pred, gt):
        # 1. Charbonnier Loss
        loss_charb = self.charbonnier_loss(pred, gt)
        
        # 2. SSIM Loss (1 - SSIM)
        ssim_val = structural_similarity_index_measure(pred, gt, data_range=1.0)
        loss_ssim = 1.0 - ssim_val
        
        # 3. Sobel Edge Loss
        loss_sobel = self.sobel_loss(pred, gt)
        
        # 4. LPIPS Loss
        pred_3c = pred.repeat(1, 3, 1, 1)
        gt_3c = gt.repeat(1, 3, 1, 1)
        pred_lpips_in = pred_3c * 2.0 - 1.0
        gt_lpips_in = gt_3c * 2.0 - 1.0
        
        loss_lpips = self.lpips_loss(pred_lpips_in, gt_lpips_in).mean()
        
        # Total Loss
        total_loss = (self.lambda_charbonnier * loss_charb + 
                      self.lambda_ssim * loss_ssim + 
                      self.lambda_sobel * loss_sobel + 
                      self.lambda_lpips * loss_lpips)
                      
        return total_loss, loss_charb.item(), ssim_val.item(), loss_sobel.item(), loss_lpips.item()

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, x, y):
        x_fft = torch.fft.fft2(x.to(torch.float32), norm="ortho")
        y_fft = torch.fft.fft2(y.to(torch.float32), norm="ortho")
        loss = self.criterion(torch.abs(x_fft), torch.abs(y_fft))
        return loss

class RestorationFreqLoss(nn.Module):
    def __init__(self, lpips_model, lambda_charbonnier=1.0, lambda_ssim=0.5, lambda_fft=0.1, lambda_lpips=0.3, device='cuda'):
        super(RestorationFreqLoss, self).__init__()
        self.lambda_charbonnier = lambda_charbonnier
        self.lambda_ssim = lambda_ssim
        self.lambda_fft = lambda_fft
        self.lambda_lpips = lambda_lpips
        
        self.charbonnier_loss = CharbonnierLoss(eps=1e-6)
        self.fft_loss = FFTLoss()
        self.lpips_loss = lpips_model
        self.device = device

    def forward(self, pred, gt):
        # 1. Charbonnier Loss
        loss_charb = self.charbonnier_loss(pred, gt)
        
        # 2. SSIM Loss (1 - SSIM)
        ssim_val = structural_similarity_index_measure(pred, gt, data_range=1.0)
        loss_ssim = 1.0 - ssim_val
        
        # 3. FFT Loss
        loss_fft = self.fft_loss(pred, gt)
        
        # 4. LPIPS Loss
        pred_3c = pred.repeat(1, 3, 1, 1)
        gt_3c = gt.repeat(1, 3, 1, 1)
        pred_lpips_in = pred_3c * 2.0 - 1.0
        gt_lpips_in = gt_3c * 2.0 - 1.0
        
        loss_lpips = self.lpips_loss(pred_lpips_in, gt_lpips_in).mean()
        
        # Total Loss
        total_loss = (self.lambda_charbonnier * loss_charb + 
                      self.lambda_ssim * loss_ssim + 
                      self.lambda_fft * loss_fft + 
                      self.lambda_lpips * loss_lpips)
                      
        return total_loss, loss_charb.item(), ssim_val.item(), loss_fft.item(), loss_lpips.item()
