from datasets import load_dataset


def return_mnist():
    train = load_dataset('mnist', split='train')
    test = load_dataset('mnist', split='test')
    return (train, test)