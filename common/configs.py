import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        default='configs/thumos14.yaml', nargs='?')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--piou', type=float, default=0.5)
    parser.add_argument('--focal_loss', type=bool)

    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        tmp = f.read()
        data = yaml.load(tmp, Loader=yaml.FullLoader)

    data['training']['piou'] = args.piou

    return data


config = get_config()
