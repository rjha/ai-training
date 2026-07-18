import string 
import argparse 
import sys 

def is_english_word(word):
    return all(char in string.ascii_letters for char in word)

def process(file_array):
    for file_name in file_array:
        
    print(file_array)
    print("processed")


def main():
    parser = argparse.ArgumentParser(description="name merge tool")
    parser.add_argument("--files", nargs="+", type=str, help="List of file paths")
    args = parser.parse_args()
    print(args)

    if len(args.files) < 2:
        print("error: we need min 2 files to merge")
        parser.print_usage()
        sys.exit(1)
    
    process(args.files)
    

    
if __name__ == "__main__":
    main()


            



