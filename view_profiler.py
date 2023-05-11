import pstats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="my_profile_stats")

args = parser.parse_args()

input_file = f'profiles/{args.name}.prof'
output_file = f'profiles/{args.name}.txt'

print("Writing results to:", output_file) 

with open(output_file, 'w') as stream:
    stats = pstats.Stats(input_file, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats()
