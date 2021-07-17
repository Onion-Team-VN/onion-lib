import click
import os

PRJ_DIR = os.getcwd()


@click.group()
def entry_point():
    pass


@click.command()
@click.option('-h', '--hidden',  'hidden',
              is_flag=True,
              help='sorted by size')
@click.option('-d', '--depth', 'depth',
              default=1,
              help='sorted by size')
def ls(hidden, depth):
    if depth > 1:
        cmd = f"du -sh * .[a-zA-Z]* |sort -h"
    else:
        cmd = f"du -hd {depth} * .[a-zA-Z]* |sort -h"
    os.system(cmd)


@click.command()
@click.option('--memory', '-m',
              is_flag=True,
              help='Open the memory note')
def open(memory):
    if memory:
        cmd = "subl /Users/hao/Projects/Personal/Experiments/Basic/basic-python-pytorch/My_Memorize"
        os.system(cmd)


entry_point.add_command(ls)
entry_point.add_command(open)
