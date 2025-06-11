import torch
import torch.nn as nn

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, patch_size=8, encoded_size=24):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256 * patch_size * patch_size, 512),
            nn.ReLU(),
            nn.Linear(512, encoded_size)
        )

    def forward(self, x):
        return self.encoder(x)

# CNN Decoder
class CNNDecoder(nn.Module):
    def __init__(self, encoded_size=24, output_channels=3, patch_size=8):
        super(CNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, 512),
            nn.ReLU(),

            nn.Linear(512, 256 * patch_size * patch_size),
            nn.ReLU(),

            nn.Unflatten(1, (256, patch_size, patch_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

# Combined Encoder-Decoder
class CNNEncoderDecoder(nn.Module):
    def __init__(self, input_channels=3, patch_size=8, encoded_size=24):
        super(CNNEncoderDecoder, self).__init__()
        self.encoder = CNNEncoder(input_channels, patch_size, encoded_size)
        self.decoder = CNNDecoder(encoded_size, input_channels, patch_size)

    def forward(self, patches):
        encoded = self.encoder(patches)
        decoded = self.decoder(encoded)
        return decoded

# Utility Functions
def images_to_patches(images, patch_size):
    """
    Convert a batch of images (b, c, w, h) into patches (b * num_patches, c, patch_size, patch_size).
    """
    b, c, w, h = images.shape
    assert w % patch_size == 0 and h % patch_size == 0, "Width and height must be divisible by patch size."
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, patch_size, patch_size)
    return patches

def patches_to_images(patches, image_size, patch_size):
    """
    Convert patches (b * num_patches, c, patch_size, patch_size) back into images (b, c, w, h).
    """
    b, c, w, h = image_size
    num_patches_w = w // patch_size
    num_patches_h = h // patch_size
    patches = patches.view(b, num_patches_w, num_patches_h, c, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    images = patches.view(b, c, w, h)
    return images

# Wrapper class to manage preloaded encoder/decoder models by resolution
class PatchCodecManager:
    def __init__(self, model_dict):
        """
        Args:
            model_dict (dict): Mapping from encoded_size (e.g., 3, 6, 12) to CNNEncoderDecoder models
        """
        self.models = model_dict

    def encode_decode(self, patches, encoded_size):
        """
        Encode and decode a batch of patches using a specific model.
        """
        if encoded_size == 196:
            return patches  # No compression for full resolution
        model = self.models.get(encoded_size)
        if model is None:
            raise ValueError(f"No model found for encoded size {encoded_size}")
        return model(patches)
