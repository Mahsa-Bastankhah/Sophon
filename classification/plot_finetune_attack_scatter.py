import json
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.cm as cm
import numpy as np

def plot_cifar_vs_imagenette(results_file, optimizer, num_samples, lr):
    with open(results_file, 'r') as f:
        data = json.load(f)

    res50_data = None
    caformer_data = None
    ## overrides the older runs
    for entry in data:
        if entry.get('optimizer') == optimizer and entry.get('sophon', {}).get('num_samples') == num_samples and entry.get('learning_rate') == lr:
            if entry.get('architecture') == 'res50':
                res50_data = entry['sophon']
            elif entry.get('architecture') == 'caformer':
                caformer_data = entry['sophon']

    if res50_data is None or caformer_data is None:
        print(f"Data for either res50 or caformer with optimizer '{optimizer}' and num_samples '{num_samples}' not found.")
        return

    # Plotting CIFAR-10 vs Imagenette accuracy for both architectures on the same plot
    plt.figure(figsize=(10, 6))
    cmap_res50 = cm.get_cmap('Reds', 30)
    cmap_caformer = cm.get_cmap('Blues', 30)

    for epoch, (cifar_acc, imagenette_acc) in enumerate(zip(res50_data['cifar10_accuracies'], res50_data['imagenette_accuracies']), start=1):
        plt.scatter(cifar_acc, imagenette_acc, color=cmap_res50(epoch / 30), alpha=0.7)

    for epoch, (cifar_acc, imagenette_acc) in enumerate(zip(caformer_data['cifar10_accuracies'], caformer_data['imagenette_accuracies']), start=1):
        plt.scatter(cifar_acc, imagenette_acc, color=cmap_caformer(epoch / 30), alpha=0.7)

    plt.xlabel('CIFAR-10 Accuracy (%)')
    plt.ylabel('Imagenette Accuracy (%)')
    plt.title(f'CIFAR-10 vs Imagenette Accuracy (Lighter : earlier epochs, darker: later epochs) (Optimizer: {optimizer}, Samples: {num_samples}, lr : {lr})')
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], marker='o', color=cmap_res50(15), markerfacecolor=cmap_res50(15), markersize=10),
                plt.Line2D([0], [0], marker='o', color=cmap_caformer(15), markerfacecolor=cmap_caformer(15), markersize=10)],
               ['ResNet50', 'CaFormer'], loc='center', bbox_to_anchor=(0.5, 0.5), ncol=1, frameon=True)
    plot_path = f'plots/cifar_vs_imagenette_{optimizer}_{num_samples}_{lr}.png'
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Saved plot: {plot_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CIFAR-10 vs Imagenette Accuracy for ResNet50 and CaFormer on the Same Plot')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer used (e.g., SGD, ADAM)')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    parser.add_argument('--num_train_samples', type=int, required=True, help='Number of training samples used')
    args = parser.parse_args()

    results_file = 'finetune_attack_results.json'
    if os.path.exists(results_file):
        plot_cifar_vs_imagenette(results_file, args.optimizer, args.num_train_samples, args.lr)
    else:
        print(f"Results file '{results_file}' not found.")