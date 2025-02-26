import torch
import torch.nn as nn
import torchvision.models as models

# Define model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=None)  # No pre-trained weights
        self.base.fc = nn.Identity()  # Remove classification layer
        
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)  # Unused in prediction
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),  # Freshness classification
            nn.Softmax(dim=1)  # Convert output to probability scores
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        freshness_output = self.block3(x)  # Extract only freshness prediction
        return freshness_output

# Function to load model
def load_model(model_path=r"C:\Users\aafiy\FreshSense-API\model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

    return model, device
