import scipy.io

import autoencoders


def get_architecture():
    layer_sizes = [('enc1', 400),
                   ('enc2', 200),
                   ('enc3', 100),
                   ('enc4', 50),
                   ('enc5', 25),
                   ('code', 6),
                   ('dec1', 25),
                   ('dec2', 50),
                   ('dec3', 100),
                   ('dec4', 200),
                   ('dec5', 400)]

    return autoencoders.get_architecture(784, layer_sizes)

def get_config():
    return autoencoders.default_config()

def run():
    try:
        obj = scipy.io.loadmat('digs3pts_1.mat')
    except:
        print("To run this script, first download www.cs.toronto.edu/~jmartens/digs3pts_1.mat to this directory.")

    X_train = obj['bdata']
    X_test = obj['bdatatest']

    config = get_config()
    arch = get_architecture()

    autoencoders.run_training(X_train, X_test, arch, config)



if __name__ == '__main__':
    run()


