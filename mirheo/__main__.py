import argparse
import mirheo

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

    compile_opt_parser = subparsers.add_parser('compile_opt', help="Get the current compile time option from its name.")
    compile_opt_parser.add_argument('name', type=str,
                                    help="The option name. The special value 'all' will list all possible options and their current values.")

    args = parser.parse_args()

    if args.version:
        print(mirheo.version)
        return

    if args.command == 'compile_opt':
        compile_opt(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
