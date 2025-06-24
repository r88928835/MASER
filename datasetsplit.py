import os
import argparse
import numpy as np
from torchvision import datasets
from numpy.random import dirichlet
import random
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Distribute dataset to multiple clients.")
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients to distribute the data to.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'], help='Dataset to use (mnist or cifar10).')
    parser.add_argument('--distribution', type=str, default='non_iid', choices=['iid', 'non_iid'], help='Distribution type (iid or non_iid).')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for Dirichlet distribution in non_iid case.')
    args = parser.parse_args()
    
    if args.num_clients <= 0:
        raise ValueError("Number of clients must be a positive integer.")
    if args.alpha <= 0:
        raise ValueError("Alpha value must be a positive number.")
    
    return args

def create_dirichlet_distribution(alpha, num_clients, num_classes, class_indices):
    proportions = dirichlet([alpha] * num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx, class_samples in enumerate(class_indices):
        class_proportions = proportions[class_idx]
        np.random.shuffle(class_samples)
        
        start_idx = 0
        for client_idx, proportion in enumerate(class_proportions):
            num_samples = int(proportion * len(class_samples))
            end_idx = start_idx + num_samples
            client_indices[client_idx].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx
        
        for client_idx in range(num_clients):
            if len(client_indices[client_idx]) == 0:
                client_indices[client_idx].append(class_samples[start_idx % len(class_samples)])
    
    return client_indices

def main():
    args = parse_arguments()
    num_clients = args.num_clients
    dataset_name = args.dataset.lower()
    distribution_type = args.distribution.lower()
    alpha = args.alpha

    base_directory = './data_distribution'
    if distribution_type == 'iid':
        save_directory = os.path.join(base_directory, dataset_name, distribution_type, f'{num_clients}clients')
    else:
        save_directory = os.path.join(base_directory, dataset_name, distribution_type, f'alpha_{float(alpha)}', f'{num_clients}clients')
    
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    if distribution_type == 'iid' and os.path.exists(save_directory):
        print(f"The {num_clients} clients {distribution_type} set for {dataset_name} is already created.")
        return
    else:
        os.makedirs(save_directory, exist_ok=True)

    # Dataset selection without transformations
    if dataset_name == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True)
        total_datapoints = 60000
    elif dataset_name == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True)
        total_datapoints = 50000
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Shuffling/Randomness
    shuffled_indices = np.random.permutation(len(trainset))
    shuffled_data = np.array(trainset.data)[shuffled_indices]
    shuffled_targets = np.array(trainset.targets)[shuffled_indices]

    total_points_per_client = total_datapoints // num_clients
    points_per_class_per_client = total_points_per_client // 10

    subsets = [[] for _ in range(num_clients)]

    if distribution_type == 'iid':
        for class_label in range(10):
            class_indices = np.where(shuffled_targets == class_label)[0]
            np.random.shuffle(class_indices)
            for client_index in range(num_clients):
                start_index = client_index * points_per_class_per_client
                end_index = (client_index + 1) * points_per_class_per_client
                client_indices = class_indices[start_index:end_index]
                if len(client_indices) < points_per_class_per_client:
                    remaining_points = points_per_class_per_client - len(client_indices)
                    additional_indices = class_indices[:remaining_points]
                    client_indices = np.concatenate((client_indices, additional_indices))
                subsets[client_index].extend(client_indices)
    elif distribution_type == 'non_iid':
        class_indices = [np.where(shuffled_targets == i)[0] for i in range(10)]
        subsets = create_dirichlet_distribution(alpha, num_clients, 10, class_indices)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    for i, subset_indices in enumerate(subsets):
        subset_data = shuffled_data[subset_indices]
        subset_targets = shuffled_targets[subset_indices]
        subset_filename = os.path.join(save_directory, f'client{i}.npz')
        np.savez(subset_filename, data=subset_data, targets=subset_targets)
        print(f"Subset {i} saved to {subset_filename}")

    for i, subset_indices in enumerate(subsets):
        subset_targets = shuffled_targets[subset_indices]
        class_counts = {class_label: np.sum(subset_targets == class_label) for class_label in range(10)}
        print(f"Client {i} - Data points per class:")
        for class_label, count in class_counts.items():
            print(f"    Class {class_label}: {count} data points")

if __name__ == '__main__':
    main()
