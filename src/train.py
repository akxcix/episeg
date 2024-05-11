import logging
import os
import torch

def train_model(model, data_loader, epochs, optimizer, criterion, device, checkpoint_path):
    model.to(device)
    model.train()

    best_loss = float('inf')
    logging.info('Starting training')
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        logging.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')
        save_checkpoint(epoch, model, optimizer, best_loss, checkpoint_path)

    logging.info('Training complete')

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch + 1} with loss {loss:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return model, optimizer, 0, float('inf')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.to(device)
    return model, optimizer, epoch, loss

def evaluate_model(model, image, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        output = torch.sigmoid(output)
        # outputs = [(output > threshold).float() for threshold in thresholds]
        # outputs_numpy = [output.cpu().numpy()[0, 0] for output in outputs]
        # return outputs_numpy
        # output = (output > 0.5).float()
        return output.cpu().numpy()[0, 0]
    
