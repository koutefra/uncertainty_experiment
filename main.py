import argparse
import experiments


def main(args):
    if args.experiment_type == 'loc':
        experiments.run_loc_experiment(args.data_dir, args.out_dir)
    elif args.experiment_type == 'unc':
        experiments.run_unc_experiment(args.data_dir, args.loc_predictors_dir, args.out_dir)
    else:
        raise ValueError('Invalid experiment type')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='output')

    parser.add_argument('--experiment_type', type=str, default='unc')
    parser.add_argument('--loc_predictors_dir', type=str, default='output/loc_experiment/models/')

    args = parser.parse_args()
    main(args)