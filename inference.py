import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from datetime import datetime
from pathlib import Path

# Import your model architecture
class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class TransformNet(nn.Module):
    """Image transformation network that learns to apply style."""
    def __init__(self):
        super(TransformNet, self).__init__()
        
        # Downsampling layers (encoder)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        
        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling layers (decoder)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, padding=4)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Encoder
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        # Decoder
        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.conv4(x)
        
        # Output in range [0, 1]
        x = torch.sigmoid(x)
        
        return x

def stylize_image(image_path, model, device, output_dir):
    """Apply style transfer to a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate stylized image
    with torch.no_grad():
        stylized = model(image_tensor)
    
    # Convert back to PIL image
    stylized = stylized.cpu().squeeze(0).permute(1, 2, 0).numpy()
    stylized = (stylized * 255).astype(np.uint8)
    stylized_image = Image.fromarray(stylized)
    
    # Resize back to original dimensions
    stylized_image = stylized_image.resize(original_size, Image.LANCZOS)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = Path(image_path).stem
    output_filename = f"{input_name}_wave_{timestamp}.jpg"
    output_path = output_dir / output_filename
    
    # Save
    stylized_image.save(output_path, quality=95)
    print(f"Saved: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Apply Wave Style Transfer to images')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--model', default='style_transfer_wave_final.pth', 
                       help='Path to trained model')
    parser.add_argument('--output', default='stylized_output', 
                       help='Output directory name')
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model = TransformNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded: {args.model}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{args.output}_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process input(s)
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        stylize_image(input_path, model, device, output_dir)
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {input_path}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for image_file in image_files:
            try:
                stylize_image(image_file, model, device, output_dir)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    else:
        print(f"Input path does not exist: {input_path}")
    
    print(f"\nDone! Check {output_dir} for stylized images.")

if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues with transforms
    main()
