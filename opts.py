import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',        
                 type=str,
                 default='mosi',
                 help=' '),
            dict(name='--dataPath',
                 default="./datasets/unaligned_50.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',     
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
           dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',        
                 default='7',
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=2,
                 type=int)
        ],

        'common': [
            dict(name='--project_name',    
                 default='ALMT',
                 type=str
                 ),
            dict(name='--seed',  # try different seeds
                 default=18,
                 type=int
                 ),
            dict(name='--models_save_root',
                 default='./checkpoint',
                 type=str
                 ),
            dict(name='--batch_size',
                 default=64,
                 type=int,
                 help=' '),
            dict(
                name='--n_threads',
                default=3,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(name='--lr',
                 type=float,
                 default=1e-4),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=200,
                type=int,
                help='Number of total epochs to run',
            )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_opts()
    print(opt.n_epochs)
