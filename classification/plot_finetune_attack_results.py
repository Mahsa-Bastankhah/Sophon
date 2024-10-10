import json
import matplotlib.pyplot as plt
import os


def plot_epoch_vs_accuracy(results_file, plots_dir):
    """
    Plot epoch vs accuracy for each specific number of training samples for both Sophon and Normal models.
    Save the plots in the specified directory.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    num_samples_dict = {}
    for entry in data:
        optimizer = entry.get('optimizer', 'SGD')
        if 'sophon' in entry:
            num_samples = entry['sophon'].get('num_samples')
            if num_samples is not None:
                if (num_samples, optimizer) not in num_samples_dict:
                    num_samples_dict[(num_samples, optimizer)] = {'sophon': entry['sophon'], 'normal': None, 'optimizer': optimizer}
                else:
                    num_samples_dict[(num_samples, optimizer)]['sophon'] = entry['sophon']
        if 'normal' in entry:
            num_samples = entry['normal'].get('num_samples')
            if num_samples is not None:
                if (num_samples, optimizer) not in num_samples_dict:
                    num_samples_dict[(num_samples, optimizer)] = {'sophon': None, 'normal': entry['normal'], 'optimizer': optimizer}
                else:
                    num_samples_dict[(num_samples, optimizer)]['normal'] = entry['normal']

    for (num_samples, optimizer), models_data in num_samples_dict.items():
        sophon_data = models_data.get('sophon')
        normal_data = models_data.get('normal')

        epochs = list(range(1, 31))

        plt.figure(figsize=(10, 6))
        if sophon_data and len(sophon_data['accuracies']) > 0:
            plt.plot(epochs, sophon_data['accuracies'], label='Fine-Tune on Sophon', color='b', linestyle='-', marker='o')
        if normal_data and len(normal_data['accuracies']) > 0:
            plt.plot(epochs, normal_data['accuracies'], label='Fine-Tune on Pretrained', color='r', linestyle='-', marker='x')

        if sophon_data or normal_data:
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Accuracy vs Epochs for {num_samples} Training Samples (Optimizer: {optimizer})')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(plots_dir, f'epoch_vs_accuracy_{num_samples}_samples_{optimizer.lower()}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f'Saved plot: {plot_path}')


def plot_final_accuracy_vs_samples(results_file, plots_dir):
    """
    Plot final accuracy vs number of training samples for both Sophon and Normal models.
    Save the plot in the specified directory.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    optimizers_dict = {}

    for entry in data:
        optimizer = entry.get('optimizer', 'SGD')
        if optimizer not in optimizers_dict:
            optimizers_dict[optimizer] = {'num_samples_list': [], 'sophon_final_accs': {}, 'normal_final_accs': {}}

        if 'sophon' in entry:
            num_samples = entry['sophon'].get('num_samples')
            if num_samples is not None:
                optimizers_dict[optimizer]['sophon_final_accs'][num_samples] = entry['sophon']['accuracies'][-1]
                if num_samples not in optimizers_dict[optimizer]['num_samples_list']:
                    optimizers_dict[optimizer]['num_samples_list'].append(num_samples)

        if 'normal' in entry:
            num_samples = entry['normal'].get('num_samples')
            if num_samples is not None:
                optimizers_dict[optimizer]['normal_final_accs'][num_samples] = entry['normal']['accuracies'][-1]
                if num_samples not in optimizers_dict[optimizer]['num_samples_list']:
                    optimizers_dict[optimizer]['num_samples_list'].append(num_samples)

    for optimizer, data_dict in optimizers_dict.items():
        num_samples_list = data_dict['num_samples_list']
        sophon_final_accs = data_dict['sophon_final_accs']
        normal_final_accs = data_dict['normal_final_accs']

        num_samples_list.sort()
        sophon_final_accs_list = [sophon_final_accs.get(num_samples) for num_samples in num_samples_list if num_samples in sophon_final_accs]
        normal_final_accs_list = [normal_final_accs.get(num_samples) for num_samples in num_samples_list if num_samples in normal_final_accs]

        plt.figure(figsize=(10, 6))
        if sophon_final_accs_list:
            plt.plot(num_samples_list[:len(sophon_final_accs_list)], sophon_final_accs_list, label='Fine-Tune on Sophon', color='b', linestyle='-', marker='o')
        if normal_final_accs_list:
            plt.plot(num_samples_list[:len(normal_final_accs_list)], normal_final_accs_list, label='Fine-Tune on Pretrained', color='r', linestyle='-', marker='x')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Final Accuracy (%)')
        plt.title(f'Final Accuracy vs Number of Training Samples (Optimizer: {optimizer})')
        plt.legend()
        plt.xscale("log")
        plt.grid(True)
        plot_path = os.path.join(plots_dir, f'final_accuracy_vs_samples_{optimizer.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Saved plot: {plot_path}')


if __name__ == '__main__':
    results_file = 'finetune_attack_results.json'
    plots_dir = 'plots'
    if os.path.exists(results_file):
        # Plot Epoch vs Accuracy for each specific number of samples
        plot_epoch_vs_accuracy(results_file, plots_dir)

        # Plot Final Accuracy vs Number of Samples
        plot_final_accuracy_vs_samples(results_file, plots_dir)
    else:
        print(f"Results file '{results_file}' not found.")