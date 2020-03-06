import argparse
import mirheo

def run(args):
    """The `mirheo run` command.

    Usage:
        run [--ranks NX NY NZ]
            ([--num-timesteps NUM]|[--num-timesteps-attr ATTR_NAME])
            snapshot_path
    """
    if bool(args.num_timesteps is not None) == bool(args.num_timesteps_attr):
        raise ValueError("Exactly one of --num-timesteps and --num-timesteps-attr must be set.")

    u = mirheo.Mirheo(args.ranks, snapshot=args.snapshot, debug_level=3, log_filename='log')
    if args.num_timesteps_attr:
        num_timesteps = int(u.getAttribute(args.num_timesteps_attr))
    else:
        num_timesteps = args.num_timesteps

    u.run(num_timesteps)

    if args.final_snapshot:
        u.saveSnapshot(args.final_snapshot)



def compile_opt(args):
    if args.name == 'all':
        options = mirheo.Utils.get_all_compile_options()
        for key, value in options.items():
            print("{} : {}".format(key, value))
    else:
        print(mirheo.Utils.get_compile_option(args.name))


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='store_true', default=False)

    subparsers = parser.add_subparsers(title="command", dest='command', help="sub-command help")

    runparser = subparsers.add_parser('run', help="Load and run a simulation from a given snapshot.")
    runparser.add_argument('--ranks', type=int, nargs=3, default=(1, 1, 1),
                           metavar=("NX", "NY", "NZ"), help="Number of ranks per dimension.")
    runparser.add_argument('--num-timesteps', type=int, metavar="NUM",
                           help="Number of timesteps.")
    runparser.add_argument('--num-timesteps-attr', type=str, metavar="ATTR_NAME",
                           help="Name of the attribute to read the number of timesteps from.")
    runparser.add_argument('--final-snapshot', type=str, metavar="PATH",
                           help="If set, a snapshot of the final state is stored to the given path.")
    runparser.add_argument('snapshot', type=str, metavar="snapshot_path",
                           help="Run the snapshot at the given path.")

    compile_opt_parser = subparsers.add_parser('compile_opt', help="Get the current compile time option from its name.")
    compile_opt_parser.add_argument('name', type=str,
                                    help="The option name. The special value 'all' will list all possible options and their current values.")

    args = parser.parse_args()

    if args.version:
        print(mirheo.version)
        return

    if args.command == 'compile_opt':
        compile_opt(args)
    elif args.command == 'run':
        run(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
