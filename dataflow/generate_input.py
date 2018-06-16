import pandas as pd

def generate():
    train = pd.read_csv('../data/train.csv', parse_dates=['activation_date'])
    train[['item_id', 'image']].to_csv('train_image.csv', index=False, header=False)

    test = pd.read_csv('../data/test.csv', parse_dates=['activation_date'])
    test[['item_id', 'image']].to_csv('test_image.csv', index=False, header=False)


if __name__ == '__main__':
    generate()