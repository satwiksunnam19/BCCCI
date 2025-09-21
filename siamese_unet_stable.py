import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import traceback

warnings.filterwarnings('ignore')

# Switched to BatchNorm2d and added Kaiming weight initialization for better training stability.

class AdaptivePadding(nn.Module):
    def __init__(self, divisor=16):
        super().__init__()
        self.divisor = divisor
    
    def forward(self, x):
        _, _, h, w = x.shape
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        return x, (pad_h, pad_w)

class RobustDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Using BatchNorm for better stability
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ProductionSiameseUNet(nn.Module):
    def __init__(self, in_channels=6, dropout_rate=0.1):
        super().__init__()
        self.adaptive_pad = AdaptivePadding()
        
        # Encoder
        self.inc = RobustDoubleConv(in_channels, 64, dropout_rate)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), RobustDoubleConv(64, 128, dropout_rate))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), RobustDoubleConv(128, 256, dropout_rate))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), RobustDoubleConv(256, 512, dropout_rate))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), RobustDoubleConv(512, 1024, dropout_rate//2))
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = RobustDoubleConv(1024, 512, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = RobustDoubleConv(512, 256, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = RobustDoubleConv(256, 128, dropout_rate)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = RobustDoubleConv(128, 64, dropout_rate)
        
        
        self.change_head = nn.Conv2d(64, 1, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pre_image, post_image):
        pre_image, pad_info = self.adaptive_pad(pre_image)
        post_image, _ = self.adaptive_pad(post_image)

        # Shared encoder
        x1_pre = self.inc(pre_image); x1_post = self.inc(post_image)
        x2_pre = self.down1(x1_pre); x2_post = self.down1(x1_post)
        x3_pre = self.down2(x2_pre); x3_post = self.down2(x2_post)
        x4_pre = self.down3(x3_pre); x4_post = self.down3(x3_post)
        x5_pre = self.down4(x4_pre); x5_post = self.down4(x4_post)
        
        fused = self.feature_fusion(torch.cat([x5_pre, x5_post], dim=1))
        
        # Decoder with skip connections from post-image
        d4 = self.up1(fused); d4 = torch.cat([x4_post, d4], dim=1); d4 = self.dec1(d4)
        d3 = self.up2(d4); d3 = torch.cat([x3_post, d3], dim=1); d3 = self.dec2(d3)
        d2 = self.up3(d3); d2 = torch.cat([x2_post, d2], dim=1); d2 = self.dec3(d2)
        d1 = self.up4(d2); d1 = torch.cat([x1_post, d1], dim=1); d1 = self.dec4(d1)
        
        if pad_info[0] > 0 or pad_info[1] > 0:
            d1 = d1[:, :, :d1.shape[2]-pad_info[0], :d1.shape[3]-pad_info[1]]
        
        # Single output for flood prediction
        logits = self.change_head(d1).squeeze(1)
        return logits

class ProductionFloodDataset(Dataset):
    def __init__(self, training_files, mode='train', patch_size=256, patches_per_image=100):
        self.training_files = training_files
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.hurricane_pairs = self._create_pairs()
        self.patches = self._generate_patches()

    def _create_pairs(self):
        # Same as before
        hurricanes = {}
        for file_info in self.training_files:
            hurricane = file_info['hurricane']
            if hurricane not in hurricanes: hurricanes[hurricane] = {'pre': [], 'post': []}
            hurricanes[hurricane][file_info['type']].append(file_info)
        pairs = []
        for hurricane, periods in hurricanes.items():
            for pre_file in periods['pre']:
                for post_file in periods['post']:
                    pairs.append({'hurricane': hurricane, 'pre_file': pre_file, 'post_file': post_file})
        return pairs

    def _load_composite(self, file_info):
        try:
            data = np.load(file_info['file_path']).astype(np.float32)
            return np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"Error loading {file_info['file_path']}: {e}")
            return None
    
    def _create_labels(self, pre_data, post_data):
        # Simplified and robust label creation
        pre_ndwi = (pre_data[1] - pre_data[3]) / (pre_data[1] + pre_data[3] + 1e-6)
        post_ndwi = (post_data[1] - post_data[3]) / (post_data[1] + post_data[3] + 1e-6)
        
        pre_water = pre_ndwi > 0.1
        post_water = post_ndwi > 0.2
        
        flood_mask = (post_water & ~pre_water).astype(np.float32)
        return flood_mask

    def _generate_patches(self):
        patches = []
        target_flood = self.patches_per_image // 2
        target_noflood = self.patches_per_image - target_flood

        for pair in tqdm(self.hurricane_pairs, desc="Creating Patches"):
            pre_composite = self._load_composite(pair['pre_file'])
            post_composite = self._load_composite(pair['post_file'])

            if pre_composite is None or post_composite is None: continue

            min_h = min(pre_composite.shape[1], post_composite.shape[1])
            min_w = min(pre_composite.shape[2], post_composite.shape[2])
            pre_composite = pre_composite[:, :min_h, :min_w]
            post_composite = post_composite[:, :min_h, :min_w]
            
            # This avoids inefficient random searching. 
            full_labels = self._create_labels(pre_composite, post_composite)
            
            # Get coordinates of all possible top-left corners for patches
            flood_y, flood_x = np.where(full_labels > 0)
            noflood_y, noflood_x = np.where(full_labels == 0)

            flood_count, noflood_count = 0, 0
            
            # Sample guaranteed flood patches
            if len(flood_y) > 0:
                for _ in range(target_flood):
                    idx = np.random.randint(0, len(flood_y))
                    cy, cx = flood_y[idx], flood_x[idx]
                    # Center the patch on the flood pixel
                    top = max(0, cy - self.patch_size // 2)
                    left = max(0, cx - self.patch_size // 2)
                    if top + self.patch_size > min_h or left + self.patch_size > min_w: continue
                    
                    pre_patch = pre_composite[:, top:top+self.patch_size, left:left+self.patch_size]
                    post_patch = post_composite[:, top:top+self.patch_size, left:left+self.patch_size]
                    label_patch = full_labels[top:top+self.patch_size, left:left+self.patch_size]
                    
                    patches.append({'pre': pre_patch, 'post': post_patch, 'labels': label_patch})
                    flood_count += 1
           
            # Sample guaranteed no-flood patches
            if len(noflood_y) > 0:
                for _ in range(target_noflood):
                    idx = np.random.randint(0, len(noflood_y))
                    cy, cx = noflood_y[idx], noflood_x[idx]
                    top = max(0, cy - self.patch_size // 2)
                    left = max(0, cx - self.patch_size // 2)
                    if top + self.patch_size > min_h or left + self.patch_size > min_w: continue

                    pre_patch = pre_composite[:, top:top+self.patch_size, left:left+self.patch_size]
                    post_patch = post_composite[:, top:top+self.patch_size, left:left+self.patch_size]
                    label_patch = full_labels[top:top+self.patch_size, left:left+self.patch_size]

                    patches.append({'pre': pre_patch, 'post': post_patch, 'labels': label_patch})
                    noflood_count += 1
        
        print(f"Generated {len(patches)} patches.")
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        # Data augmentation can be added here
        return {
            'pre': torch.FloatTensor(patch['pre']),
            'post': torch.FloatTensor(patch['post']),
            'labels': torch.FloatTensor(patch['labels']) # Use FloatTensor for BCEWithLogitsLoss
        }

# Using a combination of Focal Loss and explicit positive weighting.

class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, pos_weight_factor=10.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight_factor = pos_weight_factor

    def forward(self, logits, targets):
        # pos_weight is crucial for telling the model to care about the rare flood pixels.
        pos_weight = torch.full([1], self.pos_weight_factor, device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight)
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean()

# --- 4. TRAINER (ENHANCED) ---

class ProductionTrainer:
    def __init__(self, model, train_loader, val_loader, device='mps'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = BalancedFocalLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5, verbose=True)
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            pre = batch['pre'].to(self.device); post = batch['post'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(pre, post)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pre = batch['pre'].to(self.device); post = batch['post'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(pre, post)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.append(preds.cpu().view(-1))
                all_targets.append(labels.cpu().view(-1).long())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        f1 = f1_score(all_targets, all_preds, zero_division=0)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        
        return total_loss / len(self.val_loader), {'f1': f1, 'precision': precision, 'recall': recall}

    def train(self, epochs=15):
        best_f1 = -1
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(metrics['f1'])
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"✅ New best model saved with F1-score: {best_f1:.4f}")
        return self.history

# --- 5. EXECUTION LOGIC (CLEANED UP) ---

def run_production_training():
    print("Starting Final Fixed Hurricane Training...")
    
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    metadata_file = Path("hurricane_data/processing_metadata.json")
    if not metadata_file.exists():
        print("Metadata file not found!")
        return
        
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_files = []
    for hurricane_data in metadata.get('processed_data', []):
        for period_name, period_info in hurricane_data.get('periods', {}).items():
            file_path = period_info.get('file_path', '')
            if file_path and Path(file_path).exists():
                all_files.append({
                    'hurricane': hurricane_data.get('hurricane', 'Unknown'),
                    'file_path': file_path,
                    'type': 'pre' if 'pre' in period_name else 'post'
                })

    # **THE CRITICAL FIX FOR DATA SPLITTING**
    # Separate pre and post files to ensure a balanced split for validation.
    pre_files = [f for f in all_files if f['type'] == 'pre']
    post_files = [f for f in all_files if f['type'] == 'post']

    if not pre_files or not post_files:
        print("Error: Not enough pre or post files to create train/val splits.")
        return

    # Split pre files (80/20 split)
    pre_split_idx = max(1, int(0.8 * len(pre_files)))
    train_pre = pre_files[:pre_split_idx]
    val_pre = pre_files[pre_split_idx:]
    if not val_pre and len(pre_files) > 1: # Ensure val has at least one file
        val_pre = [pre_files[-1]]
        train_pre = pre_files[:-1]

    # Split post files (80/20 split)
    post_split_idx = max(1, int(0.8 * len(post_files)))
    train_post = post_files[:post_split_idx]
    val_post = post_files[post_split_idx:]
    if not val_post and len(post_files) > 1: # Ensure val has at least one file
        val_post = [post_files[-1]]
        train_post = post_files[:-1]

    train_files = train_pre + train_post
    val_files = val_pre + val_post
    
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    train_dataset = ProductionFloodDataset(train_files, patches_per_image=100)
    val_dataset = ProductionFloodDataset(val_files, patches_per_image=50)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Failed to generate patches. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = ProductionSiameseUNet()
    trainer = ProductionTrainer(model, train_loader, val_loader, device=device)
    
    history = trainer.train(epochs=20) # Train for more epochs
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='F1'); plt.plot(history['val_precision'], label='Precision'); plt.plot(history['val_recall'], label='Recall')
    plt.legend(); plt.title('Metrics'); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    run_production_training()

