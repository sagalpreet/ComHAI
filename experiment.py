from dependencies import *
from combiner import *
import random
import argparse

accuracies = None
NUM_HUMANS = None

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--accuracies', nargs='+', type=float)
    p.add_argument('--model_names', nargs='+', type=str)
    args = p.parse_args()

    accuracies = args.accuracies
    accuracies.sort(reverse=1)
    num_humans = len(accuracies)
    model_names = args.model_names

    assert max(accuracies) <= 1, "Accuracy of the agent must be less than or equal to 1"
    assert min(accuracies) >= 0, "Accuracy of the agent must be less than or equal to 1"
    assert num_humans > 0, "Number of agents must be greater than 0"

    return accuracies, num_humans, model_names

def load_CIFAR10H(model_name):
    """ Loads the CIFAR-10H predictions (human and model) and true labels.
    """
    dirname = PROJECT_ROOT

    data_path = os.path.join(dirname, f'dataset/{model_name}.csv')
    try:
        data = np.genfromtxt(data_path, delimiter=',')
    except:
        raise ValueError(f"Invalid model name. Model with name '{model_name}' does not exist")

    true_labels = data[:, 0]
    human_counts = data[:, 1:11]
    model_probs = data[:, 11:]

    true_labels = true_labels.astype(int)

    return human_counts, model_probs, true_labels

def simulate_humans(human_counts, y_true, accuracy_list, seed=0):
    rng = np.random.default_rng(seed)
    human_labels = []

    assert len(human_counts) == len(y_true), "Size mismatch"

    i = -1

    for data_point in human_counts:
        i += 1
        labels = []
        for accuracy in accuracy_list:
            if (rng.random() < accuracy):
                labels.append(y_true[i])
            else:
                prob = data_point
                prob[y_true] = 0
                if (np.sum(prob) == 0):
                    prob = np.ones(prob.shape)
                    prob[y_true[i]] = 0
                prob /= np.sum(prob)
                labels.append(rng.choice(range(len(data_point)), p = prob))
                
        human_labels.append(labels)
    
    return np.array(human_labels)

def get_acc(y_pred, y_true):
    """ Computes the accuracy of predictions.
    If y_pred is 2D, it is assumed that it is a matrix of scores (e.g. probabilities) of shape (n_samples, n_classes)
    """
    if y_pred.ndim == 1:
        return np.mean(y_pred == y_true)
    print("Invalid Arguments")

def main():
    global accuracies, NUM_HUMANS
    accuracies, NUM_HUMANS, model_names = parse_arguments()

    n_runs = 10
    test_sizes = [0.999, 0.99, 0.9, 0.0001]

    out_fpath = './output/'
    os.makedirs(out_fpath, exist_ok=True)

    for test_size in test_sizes:

        for model_name in tqdm(model_names, desc='Models', leave=True):
            # Specify output files
            output_file_acc = out_fpath + f'{model_name}_accuracy_{str(accuracies)}_{int((1-test_size)*10000)}'

            # Load data
            human_counts, model_probs, y_true = load_CIFAR10H(model_name)

            # Generate human output from human counts through simulation
            y_h = simulate_humans(human_counts, y_true, accuracies)

            POLICIES = [
                ('single_best_policy', single_best_policy, False),
                ('mode_policy', mode_policy, False),
                ('weighted_mode_policy', weighted_mode_policy, False),
                ('select_all_policy', select_all_policy, False),
                ('random', random_policy, False),
                ('lb_best_policy', lb_best_policy, True),
                ('pseudo_lb_best_policy_overloaded', pseudo_lb_best_policy_overloaded, False)
            ]

            acc_data = []
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                seed = random.randint(1, 1000)
                # Train/test split
                y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_probs, y_true, test_size=test_size, random_state=i * seed)

                # Test over entire dataset
                y_h_te = y_h
                model_probs_te = model_probs
                y_true_te = y_true

                acc_h = get_acc(y_h_te[:, 0], y_true_te) # considering the accuracy of the best human only
                acc_m = get_acc(np.argmax(model_probs_te, axis=1), y_true_te)

                _acc_data = [acc_h, acc_m]
                
                add_predictions("True Labels", y_true_te)

                combiner = MAPOracleCombiner()

                combiner.fit(model_probs_tr, y_h_tr, y_true_tr, NUM_HUMANS)

                for policy_name, policy, use_true_labels in POLICIES:

                    humans = policy(combiner, y_h_te, y_true_te if use_true_labels else None, np.argmax(model_probs_te, axis=1), NUM_HUMANS, model_probs_te.shape[1])
                    
                    y_comb_te = combiner.combine(model_probs_te, y_h_te, humans)

                    acc_comb = get_acc(y_comb_te, y_true_te)
                    _acc_data.append(acc_comb)

                acc_data += [_acc_data]

            header_acc = ['human', 'model'] + [policy_name for policy_name, _, _ in POLICIES]
            with open(f'{output_file_acc}_{i}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_acc)
                writer.writerows(acc_data)

main()