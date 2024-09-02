import matplotlib.pyplot as plt
import re
import os

def extract_values(file_path):
    # Pattern to match the relevant log lines and capture values
    pattern = r"epoch:(\d+)/\d+.*?loss_rec:([\d\.e\-]+).*?loss_geo:([\d\.e\-]+).*?loss_dis:([\d\.e\-]+).*?psnr:([\d\.]+)"
    
    # Dictionary to store the results with epochs as keys
    results = {}
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss_rec = float(match.group(2))
                loss_geo = float(match.group(3))
                loss_dis = float(match.group(4))
                psnr = float(match.group(5))
                
                # Store the extracted values in the dictionary
                results[epoch] = {
                    'loss_rec': loss_rec,
                    'loss_geo': loss_geo,
                    'loss_dis': loss_dis,
                    'psnr': psnr
                }
    
    return results


model_type = 'IFRNet_S_T2'

#IFRNet_S --> 2.80M
#IFRNet_S_T1 --> 2.21M
#IFRNet_S_T2 --> 1.25M
num_params = 1.25

date = '2024-04-21 14-24-48'
log_file_path = os.path.join('checkpoint', model_type, date, 'train.log')

results = extract_values(log_file_path)

# Extract the values for plotting
epochs = list(results.keys())
loss_rec = [results[epoch]['loss_rec'] for epoch in epochs]
loss_geo = [results[epoch]['loss_geo'] for epoch in epochs]
loss_dis = [results[epoch]['loss_dis'] for epoch in epochs]
psnr = [results[epoch]['psnr'] for epoch in epochs]

# Plot the values
plt.figure(figsize=(12, 6))
plt.plot(epochs, loss_rec, label='Reconstruction Loss')
plt.plot(epochs, loss_dis, label='Discriminator Loss')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.title(f'Model:{model_type} {num_params}M Training Losses')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, loss_geo, label='Geometric Consistency Loss')
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.title(f'Model:{model_type} {num_params}M Training Losses')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(epochs, psnr, label='PSNR')
plt.xlabel('Epochs')
plt.ylabel('PSNR')
plt.title(f'Model:{model_type} {num_params}M PSNR')
plt.legend()
plt.grid()
plt.show()

print("Max PSNR: ", max(psnr))