import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

def visualize_superpixels(image, segments):
    """Visualize superpixel segmentation results"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mark_boundaries(image, segments))
    ax.set_title('Superpixel Segmentation')
    ax.axis('off')
    return fig

def save_results(results, filename='results.txt'):
    """Save training results to file"""
    with open(filename, 'w') as f:
        for epoch, (loss, acc) in enumerate(results):
            f.write(f'Epoch {epoch+1}: Loss={loss:.4f}, Acc={acc:.2f}%\n')

def plot_training_curve(results):
    """Plot training loss and accuracy curves"""
    losses = [r[0] for r in results]
    accs = [r[1] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    ax2.plot(accs)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    
    return fig
