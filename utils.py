
import yaml
import argparse


def get_config(filename):
  parser = argparse.ArgumentParser(description='Default parse')

  with open(filename, 'r') as f:
    cfg = yaml.load(f)

  parser = add_cfg_to_parser(cfg, parser)

  args = parser.parse_args()
  return args


def add_cfg_to_parser(cfg, parser, root=''):
  for k, v in cfg.items():
    rk = k if root == '' else root + '.' + k
    if isinstance(v, dict):
      parser = add_cfg_to_parser(v, parser, root=rk)
    else:
      parser.add_argument("--" + rk, type=type(v), default=v)

  return parser
