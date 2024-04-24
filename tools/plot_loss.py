import matplotlib.pyplot as plt
import pickle

with open('loss_histories.pkl', 'rb') as f:
    loss_data = pickle.load(f)
    train_losses = loss_data['train_losses']
    valid_losses = loss_data['valid_losses']


# Assuming 'train_losses' and 'valid_losses' are loaded
plt.figure(figsize=(10, 5))
plt.plot([loss[0] for loss in train_losses], [loss[1] for loss in train_losses], label='Train Loss')
if valid_losses:
    plt.plot([loss[0] for loss in valid_losses], [loss[1] for loss in valid_losses], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.show()
