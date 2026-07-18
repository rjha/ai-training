import string 
import argparse 
import os
import pandas

_names = set() 

def is_english_word(word):
    return all(char in string.ascii_letters for char in word)

def process_line(line):
    if not line:
        return 
    
    tokens = str(line).strip().split(',')
    if tokens:
        raw_name = tokens[0]
        if raw_name:
            names =  raw_name.split(' ')
            for name in names:
                _names.add(name.lower())

def process_file(file_name):
    if not os.path.exists(file_name):
        raise ValueError("check input file path: " + file_name)
    with open(file_name) as f:
        for line in f:
            process_line(line)
            
def main():
    parser = argparse.ArgumentParser(description="raw name data processing tool")
    parser.add_argument("--files", nargs="+", type=str, help="List of file paths", required=True)
    args = parser.parse_args()
    print(args)

    for file_name in args.files:
        process_file(file_name)
    
    counter = 0
    for name in _names:
        print(name)
        counter = counter + 1

    print("total names: {0}".format(counter))


if __name__ == "__main__":
    main()
