from importlib.machinery import SourceFileLoader
from argparse import ArgumentParser, MetavarTypeHelpFormatter

import inspect


parser = ArgumentParser(
    prog='python -m kubemo',
    description='ML model deployment made simple.',
    formatter_class=MetavarTypeHelpFormatter,
)
parser.add_argument('--name', type=str, help='specify the name of your inference class')
parser.add_argument('--path', type=str, help='specify the path to your kubemo script')
parser.add_argument('--model', type=str, help='specify the path to your bundled model')
parser.add_argument('--address', type=str, help='specify an address to serve')
args = parser.parse_args()

print(args.name, args.path)

