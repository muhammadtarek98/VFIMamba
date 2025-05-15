import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from Trainer_finetune import Model
from dataset import CustomDataset  # Replace with your dataset class
from torch.nn import MSELoss  # Replace with your loss function if needed

# Configuration
local_rank = -1  # Set to GPU ID if using distributed training
epochs = 10
batch_size = 16
learning_rate = 1e-4

# Initialize model
model = Model(local_rank)
model.train()

# Load dataset
dataset = CustomDataset()  # Replace with your dataset class
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = Adam(model.net.parameters(), lr=learning_rate)
criterion = MSELoss()  # Replace with your loss function if needed

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in data_loader:
        inputs, targets = batch  # inputs: (frame_1, frame_2), targets: (frame_3, ..., frame_8)
        frame_1, frame_2 = inputs
        frame_1, frame_2 = frame_1.cuda(), frame_2.cuda()
        targets = [t.cuda() for t in targets]  # Ensure all target frames are on GPU

        optimizer.zero_grad()

        # Predict 6 consecutive frames
        predicted_frames = []
        current_frame = frame_1
        for i in range(6):
            next_frame = model.net(torch.cat((current_frame, frame_2), dim=1))  # Concatenate inputs
            predicted_frames.append(next_frame)
            current_frame = next_frame

        # Compute loss
        loss = sum(criterion(pred, target) for pred, target in zip(predicted_frames, targets))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.net.state_dict(), f"ckpt/{model.name}_trained.pkl")
