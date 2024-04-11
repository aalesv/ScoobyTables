#Make xml file created by ScoobyTable more readable
#Add suffixes '_A', '_B' etc or '_1', '_2' etc to same table names
import argparse
from collections import Counter
from lxml import etree as ET

VERSION=2024.0411

def char_count(count):
    """
    Characters generator. Starts from 'A' and executes for count times.
    """
    start = 'A'
    for char in range(ord(start), ord(start)+count):
        yield chr(char)

def num_count(count):
    """
    Numbers generator. Starts from 1 and executes for count times.
    """
    start = 1
    for i in range (start, start+count):
        yield i

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Add suffixes "_A", "_B" etc or "_1", "_2" etc to same table names.',
    epilog='For better results please clean and rename tables first!'
)
parserGroupOptions = parser.add_argument(
    '-i', '--input',
    metavar='<filename>',
    help='Input filename'
)
parserGroupOptions = parser.add_argument(
    '-o', '--output',
    metavar='<filename>',
    help='Output filename. stdout if not specified'
)
parserGroupOptions = parser.add_argument(
    '-F', '--word-separator',
    metavar='<symbol>',
    help='Word separator',
    default='_'
)
parserGroupOptions = parser.add_argument(
    '--numeric-suffix',
    action='store_true',
    help='Suffix is numeric'
)
parserGroupOptions = parser.add_argument(
    '--version',
    action='version',
    help='Print version number.',
    version=f'{VERSION}'
)

args = parser.parse_args()

input_file  = args.input
output_file = args.output
word_separator=args.word_separator

OUTPUT_TO_STDOUT = output_file is None

parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
xml_root = ET.parse(input_file, parser)
#Array for all table names
tables = []
xml_all_tables = xml_root.xpath('./table2D | ./table3D')
for t in xml_all_tables:
    tn = t.get('name')
    if tn != '':
        tables.append(tn)

#Dictionary. Key = table name, value = count of appearance
tables_dict_count = Counter(tables)
#List of tables that appears more than once
tables_many = list(i for i in tables_dict_count if tables_dict_count[i] > 1)
#Dictionary of generators
tables_many_gens = {}
#Set suffix generator function based on CLI flags
suffix_generator = num_count if args.numeric_suffix else char_count
#Set to every unique name a generator that will generate suffix
for i in tables_many:
    tables_many_gens[i] = suffix_generator(tables_dict_count[i])

#Check every table
for i in xml_all_tables:
    name = i.get('name')
    #If this table appears more than one time
    if name in tables_many:
        #Generate next letter suffix
        next_suffix = next(tables_many_gens[name])
        i.set('name', f'{name}{word_separator}{next_suffix}')

xml_root.addprevious(ET.Comment(f'modified by beautify_xml.py v{VERSION}'))
tree = ET.ElementTree()
tree._setroot(xml_root)
if OUTPUT_TO_STDOUT:
    print(ET.tostring(tree, pretty_print=True, encoding="utf-8", xml_declaration=True).decode())
else:
    tree.write(output_file, pretty_print=True, encoding="utf-8", xml_declaration=True)
