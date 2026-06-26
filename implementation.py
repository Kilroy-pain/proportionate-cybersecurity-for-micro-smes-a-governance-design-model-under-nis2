import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CyberSecurityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CyberSecurityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def generate_dummy_data(num_samples, input_dim, num_classes):
    X = np.random.rand(num_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    # Hyperparameters
    input_dim = 7  # Seven dimensions from the paper
    hidden_dim = 16
    output_dim = 2  # Binary classification: secure or insecure
    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_dim, output_dim)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CyberSecurityModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate the model
    train_model(model, criterion, optimizer, train_loader, num_epochs)
    evaluate_model(model, test_loader)