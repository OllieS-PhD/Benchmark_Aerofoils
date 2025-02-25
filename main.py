import argparse





def get_args():
    # Create argument parser for 
    parser = argparse.ArgumentParser(description='Args for graph prediction')
    parser.add_argument('-num_epochs', type=int, default=2, help='epochs')
    parser.add_argument('-batch', type=int, default=8, help='batch size')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-l_num', type=int, default=3, help='layer num')
    parser.add_argument('-h_dim', type=int, default=512, help='hidden dim')
    parser.add_argument('-l_dim', type=int, default=48, help='layer dim')
    parser.add_argument('-act', type=str, default='ReLU', help='activation function')
    parser.add_argument('-drop_n', type=float, default=0.3, help='drop net')
    parser.add_argument('-drop_c', type=float, default=0.2, help='drop output')
    parser.add_argument('-ks', nargs='+', type=float, default='0.9 0.8 0.7')
    args, _ = parser.parse_known_args()
    return args


def run():
    args = get_args()



if __name__ == "__main__":
    run()