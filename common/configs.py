import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        default='configs/thumos14.yaml', nargs='?')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--piou', type=float, default=0.5)
    parser.add_argument('--focal_loss', type=bool)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--lw', type=float, default=10.0)
    parser.add_argument('--cw', type=float, default=1)
    parser.add_argument('--resume', type=int, default=2)

    parser.add_argument('--nms_thresh', type=float)
    parser.add_argument('--nms_sigma', type=float)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--output_json', type=str)

    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        tmp = f.read()
        data = yaml.load(tmp, Loader=yaml.FullLoader)

    data['training']['piou'] = args.piou
    if args.checkpoint_path is not None:
        data['training']['checkpoint_path'] = args.checkpoint_path
        data['testing']['checkpoint_path'] = args.checkpoint_path
    data['training']['lw'] = args.lw
    data['training']['cw'] = args.cw
    data['training']['resume'] = args.resume
    if args.seed is not None:
        data['training']['random_seed'] = args.seed
    if args.nms_thresh is not None:
        data['testing']['nms_thresh'] = args.nms_thresh
    if args.nms_sigma is not None:
        data['testing']['nms_sigma'] = args.nms_sigma
    if args.top_k is not None:
        data['testing']['top_k'] = args.top_k
    if args.output_json is not None:
        data['testing']['output_json'] = args.output_json
    return data


config = get_config()
