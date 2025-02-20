import torchvision.datasets
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load MNIST Training Dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Step 2: Model Definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=64):
        super(Autoencoder, self).__init__()
        
        # Encoder: [batch_size, 784] -> [batch_size, 256] -> [batch_size, 64]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 784 -> 256
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),  # 256 -> 64
            nn.ReLU()
        )
        
        # Decoder: [batch_size, 64] -> [batch_size, 256] -> [batch_size, 784]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # 64 -> 256
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # 256 -> 784
            nn.Sigmoid()  # output in range [0,1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 3: Training Loop
epochs = 5
for epoch in range(epochs):
    for images, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 4: Inference and Applications
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Show original vs reconstructed images
    sample_images, _ = next(iter(train_loader))  # [batch_size, 784]
    sample_images = sample_images.to(device)
    reconstructed_images = model(sample_images)

# Reshape images back to 28x28 for visualization
sample_images = sample_images.cpu().view(-1, 28, 28)
reconstructed_images = reconstructed_images.cpu().view(-1, 28, 28)

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(sample_images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')
    axes[1, i].imshow(reconstructed_images[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Reconstructed')
plt.show()

# Application 1: Anomaly Detection
# Load MNIST Test Data
mnist_test = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms, download=True
)

mnist_images = mnist_test.data.float() / 255.0
mnist_images = mnist_images.view(-1, 784).to(device)

with torch.no_grad():
    reconstructed_mnist = model(mnist_images)
    mnist_error = torch.mean((mnist_images - reconstructed_mnist) ** 2, dim=1)

mnist_sorted_indices = torch.argsort(mnist_error, descending=True)
mnist_top_images = mnist_images[mnist_sorted_indices].cpu().view(-1, 28, 28)
mnist_top_recons = reconstructed_mnist[mnist_sorted_indices].cpu().view(-1, 28, 28)
mnist_top_errors = mnist_error[mnist_sorted_indices].cpu()

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(mnist_top_images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')
    axes[1, i].imshow(mnist_top_recons[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Reconstructed')
plt.show()

# Denoising Autoencoder Experiment
# Add Gaussian noise to the sample images
noisy_images = sample_images + 0.3 * torch.rand_like(sample_images)
noisy_images = torch.clamp(noisy_images, min=0.0, max=1.0)

# Flatten
with torch.no_grad():
    noisy_images_flat = noisy_images.view(-1, 784).to(device)
    denoised_outputs = model(noisy_images_flat)

denoised_outputs = denoised_outputs.cpu()

fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    axes[0, i].imshow(noisy_images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Noisy')
    axes[1, i].imshow(denoised_outputs[i].view(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Denoised')
plt.show()