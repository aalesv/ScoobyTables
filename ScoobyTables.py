import argparse
import glob
import re
import struct
from lxml import etree as ET
import numpy as np
import pyarrow
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle
from enum import Enum, auto

VERSION = '2024.0311'

class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value: str):
        for member in cls:
            if member.lower() == value.lower():
                return member
        return None

class DataFormat(CaseInsensitiveEnum):
    xml = 'xml'
    csv = 'csv'

class ArgSetNonDefaultAttr(argparse.Action):
    """
    If argument with default value was specified on command line, 
    a new attribute named <argument_name>_nondefault set
    """
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest+'_nondefault', True)

def normalizeDataFrame(dataFrame):
    """
    Normalizes DataFrame
    """
    #Convert to float
    #all except 'name' column
    #dataFrame.loc[:, dataFrame.columns != 'name'] = dataFrame.loc[:, dataFrame.columns != 'name'].astype(float)
    #Last column is string
    header = dataFrame.columns
    for i in header[:len(header)-1]:
        dataFrame[i] = dataFrame[i].astype(float)

    #Convert to distance from start of data area
    dataFrame['distance'] = dataFrame['storageaddress']
    dataFrame.drop(['storageaddress'], axis=1, inplace=True)
    min = dataFrame['distance'].min()
    max = dataFrame['distance'].max() - min
    dataFrame['distance'] -= min
    dataFrame['distance'] /= max
    dataFrame['distance'] *= 100
    dataFrame['x_len'] *= 5
    dataFrame['y_len'] *= 5
    #Calculate average
    dataFrame['x_avg'] = (dataFrame['x_min'] + dataFrame['x_max']) / 2
    dataFrame['y_avg'] = (dataFrame['y_min'] + dataFrame['y_max']) / 2
    dataFrame.drop(['x_min','x_max','y_min','y_max'], axis=1, inplace=True)
    #Increase scale of multiplier to make it more significant
    dataFrame['multiplier'] *= 10000
    #
    dataFrame['data_type'] *= 25
    return dataFrame

def createDataFrame(array, nColumns, header):
    """
    Create dataframe from array 'array'. 
    Returns Dataframe which will contain 'nColumns' columns and 'header' header.
    """
    array = array.reshape(-1, nColumns)
    dataFrame = pd.DataFrame(array, columns=header)
    return dataFrame

def createAndNormalizeDataFrame(array, nColumns, header):
    """
    Create and normalize dataframe from array 'array'. 
    Returns Dataframe which will contain 'nColumns' columns and 'header' header.
    """
    df = createDataFrame(array, nColumns, header)
    df = normalizeDataFrame(df)
    return df

def cleanDataFrameRows(dataFrame):
    """Delete strings with empty 'name' column values"""
    dataFrame = dataFrame.query('name != ""')
    return dataFrame.reset_index(drop=True)

def cleanDataFrameColumns(dataFrame):
    """Delete columns that are not in HEADER"""
    dataFrame.drop([c for c in dataFrame.columns if not(c in HEADER)], axis=1, inplace=True)
    return dataFrame

def loadXmlFromDirNormalized(dirname):
    """
    Load dataset from ROM binary files located in 'dirname'.\n
    XML must be in ScoobyRom format. Returns DataFrame.
    """
    files = glob.glob(f'{dirname}/*.xml')
    dataFrame = pd.DataFrame()
    for f in files:
        binFileName = re.sub('\.xml$', '.bin', f)
        (t, nColumns, header) = parseOneXmlFile(f, binFileName)
        tempDataFrame = createAndNormalizeDataFrame(t, nColumns, header)
        frames = [dataFrame, tempDataFrame]
        dataFrame = pd.concat(frames, ignore_index=True)
    return dataFrame

def parseOneXmlFile(xmlFileName, binFileName):
    """
    Parse one ROM file. XML must be in ScoobyRom format.\n
    Returns: (dataset array, number of columns, headers name array of string).
    """
    #Do not filter XML comments
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(xmlFileName, parser)
    #Open for read in binary mode
    binFile = open(binFileName, 'rb')
    ds = np.array([])
    for child in root:
        if child.tag == 'table2D' or child.tag == 'table3D':
            #Parsing comment
            comments=[]
            for subchild in child:
                if "function Comment" in str(subchild.tag): 
                    comments.append(subchild.text)
            #Expecting comment with min and max values in form
            #<!-- -30 to 55 -->
            x_min, _ ,x_max = comments[0].split(maxsplit=3)
            #For 2D tables second comment differs
            s = comments[1].split(maxsplit=3)
            y_min = s[0]
            y_max = s[2]
            #Convert 'name' attribute
            #Convert 'Cranking_A' or 'Cranking_1' to 'cranking'
            name = re.sub('_[a-z0-9]$', '', child.attrib.get('name').lower())
            storageaddress = int(child.attrib.get('storageaddress'), 16)
            #Go to table structure location
            binFile.seek(storageaddress)
            multiplier = 0
            offset = 0
            if child.tag == 'table2D':
                #2D table structure
                # x_len:          .data.w
                # data_type:      .data.w
                # x_ptr:          .data.l
                # data_ptr:       .data.l
                # multiplier:     .float
                # offset:         .float
                #
                #Usually if data_type==0, there's no multiplier and offset fields.
                x_len = int.from_bytes(binFile.read(2))
                y_len = y_min = y_max = 0
                
                data_type = int.from_bytes(binFile.read(2))
                if data_type > 0:
                    binFile.seek(8, 1)
                    #Read IEEE754 float little endian
                    (multiplier,) = struct.unpack('>f', binFile.read(4))
                    (offset,) = struct.unpack('>f', binFile.read(4))

            if child.tag == 'table3D':
                #3D table structure
                # x_len:          .data.w
                # y_len:          .data.w
                # x_axis_ptr:     .data.l
                # y_axis_ptr:     .data.l
                # data_ptr:       .data.l
                # data_type:      .data.l
                # multiplier:     .float
                # offset:         .float
                #
                #Usually if data_type==0, there's no multiplier and offset fields.
                x_len = int.from_bytes(binFile.read(2))
                y_len = int.from_bytes(binFile.read(2))
                #seek +12 bytes from here
                binFile.seek(12, 1)
                data_type = int.from_bytes(binFile.read(4))
                if data_type > 0:
                    #Read IEEE754 float little endian
                    (multiplier,) = struct.unpack('>f', binFile.read(4))
                    (offset,) = struct.unpack('>f', binFile.read(4))

            #Convert data_type column to machine-readable format
            match data_type:
                case 0x400:
                    #2D uint8
                    data_type = 1
                case 0x800:
                    #2D uint16
                    data_type = 2
                case 0x4000000:
                    #3D uint8
                    data_type = 3
                case 0x8000000:
                    #3D uint16
                    data_type = 4
            #Header. Place 'name' last!
            h = HEADER
            t = np.array([ storageaddress,  x_len,  x_min,  x_max,  y_len,  y_min,  y_max,  data_type,  multiplier,  offset,  name])
            (ncolumns,) = t.shape
            ds = np.append(ds, t)
    return ds, ncolumns, h

def loadCsvFromDirNormalized(dirname):
    """
    Load dataset from several CSV files located in dirname directory.\n
    CSV can be exported from ScoobyRom. Returns DataFrame.
    """
    files = glob.glob(f'{dirname}/*.csv')
    dataFrame = pd.DataFrame()
    for f in files:
        tempDataFrame = parseOneCsvFile(f)
        tempDataFrame = normalizeDataFrame(tempDataFrame)
        frames = [dataFrame, tempDataFrame]
        dataFrame = pd.concat(frames, ignore_index=True)

    return dataFrame

def readRawCsvFile(filename):
    """
    Read CSV file. Return it as is, without any transformations
    """
    dataFrame = pd.read_csv(filename, keep_default_na=False)
    return dataFrame

def parseOneCsvFile(filename):
    """
    Load dataset from filename in CSV format, clean columns and convert data_type to machine-readable format. 
    CSV can be exported from ScoobyRom. Returns DataFrame.
    """
    dataFrame = readRawCsvFile(filename)
    dataFrame = cleanDataFrameColumns(dataFrame)
    #Convert data_type column to machine-readable format
    dataFrame.loc[dataFrame['multiplier'] == 0, 'data_type']      = 0
    dataFrame.loc[(dataFrame['y_len'] == 0) &
                  (dataFrame['data_type'] == UINT8), 'data_type'] = 1
    dataFrame.loc[(dataFrame['y_len'] == 0) &
                  (dataFrame['data_type'] == UINT16),'data_type'] = 2
    dataFrame.loc[(dataFrame['y_len'] != 0) &
                  (dataFrame['data_type'] == UINT8), 'data_type'] = 3
    dataFrame.loc[(dataFrame['y_len'] != 0) &
                  (dataFrame['data_type'] == UINT16),'data_type'] = 4
    #Convert 'name' column
    #Convert 'Cranking_A' or 'Cranking_1' to 'cranking'
    dataFrame['name'] = dataFrame['name'].str.lower()
    dataFrame['name'] = dataFrame['name'].replace(regex=r'(.*)_[a-z0-9]$', value=r'\1')
    return dataFrame

def trainKNN(x, y):
    """
    Train KNN model. x - data dataframe, y - names dataframe. Returns trained KNN model
    """
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x, y)
    return knn

def dumpModel(model, filename):
    """Dump model to file"""
    outFile = open(filename, 'wb')
    pickle.dump(model, outFile)
    outFile.close()

def loadModel(filename):
    """Load model from file"""
    inFile = open(filename, 'rb')
    model = pickle.load(inFile)
    inFile.close()
    return model

def print_console_verbose(string):
    """Print string to console if in verbose mode"""
    global args
    if args.verbose:
        print(string)

def getCorrectedXMLtree(filename, dataFrame):
    """
    Take ScoobyRom def from file. 
    Replace 'name' attribute in all tags
    to corresponing values from dataFrame['name']
    where 'storageaddress' attribute is equal to dataFrame['storageaddress'] value.
    """
    array = dataFrame[['storageaddress', 'name']].to_numpy()
    #Do not filter XML comments
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.ElementTree()
    root = ET.parse(xmlFileName, parser)
    for a in array:
        #Convert to int from hex string
        addr = int(a[0], base=16)
        #Find all tags where 'storageaddress'==addr (in hexadecimal representation)
        t = root.find(f'.//*[@storageaddress="0x{addr:X}"]')
        #Replace 'name' attribute
        t.set('name', a[1])
    root.addprevious(ET.Comment(f'modified by ScoobyTables v{VERSION}'))
    tree._setroot(root)
    return tree

def createElement(tagName, attrNameValueTuplesArray):
    """
    Create XML element. Tag = tagName, 
    attributes and values = [('attr1', 'value1'),('attr2', 'value2'), ...]
    Returns: '<tagName attr1="value1" attr2="value2" ... />'
    """
    element = ET.Element(tagName)
    for t in attrNameValueTuplesArray:
        (attrName, attrValue) = t
        element.set(attrName, attrValue)
    return element

def createTable2DXml(row):
    """Create 2D table XML element from dataframe row. Returns Element."""
    attributesArray = [
        ('category', row['category']),
        ('name', row['name']),
        ('storageaddress', row['storageaddress'])
    ]
    table2D = createElement('table2D', attributesArray)
    
    attributesArray = [
        ('storageaddress', row['axis_x_storageaddress']),
        ('name', row['name_x']),
        ('unit', row['unit_x'])
    ]
    axisX = createElement('axisX', attributesArray)
    commentX = ET.Comment(f' {row["x_min"]} to {row["x_max"]} ')

    attributesArray = [
        ('storageaddress', row['axis_z_storageaddress']),
        ('unit', row['unit_z']),
        ('storagetype', row['data_type'])
    ]
    values = createElement('values', attributesArray)
    #This is needed to compatibility, ScoobyRom ignores this
    commentV = ET.Comment(' min: Unknown  max: Unknown  average: Unknown ')

    table2D.append(axisX)
    table2D.append(values)
    table2D.insert(0, commentX)
    table2D.insert(2, commentV)
    return table2D

def createTable3DXml(row):
    """Create 3D table XML element from dataframe row. Returns Element."""
    attributesArray = [
        ('category', row['category']),
        ('name', row['name']),
        ('storageaddress', row['storageaddress'])
    ]
    table3D = createElement('table3D', attributesArray)
    
    attributesArray = [
        ('storageaddress', row['axis_x_storageaddress']),
        ('name', row['name_x']),
        ('unit', row['unit_x'])
    ]
    axisX = createElement('axisX', attributesArray)
    commentX = ET.Comment(f' {row["x_min"]} to {row["x_max"]} ')

    attributesArray = [
        ('storageaddress', row['axis_y_storageaddress']),
        ('name', row['name_y']),
        ('unit', row['unit_y'])
    ]
    axisY = createElement('axisX', attributesArray)
    commentY = ET.Comment(f' {row["y_min"]} to {row["y_max"]} ')

    attributesArray = [
        ('storageaddress', row['axis_z_storageaddress']),
        ('unit', row['unit_z']),
        ('storagetype', row['data_type'])
    ]
    values = createElement('values', attributesArray)

    table3D.append(axisX)
    table3D.append(axisY)
    table3D.append(values)
    table3D.insert(0, commentX)
    table3D.insert(2, commentY)
    return table3D

def createXmlFromCsv(dataFrame):
    """
    Create minimum valid ScoobyRom XML definitions file. 
    Returns ElementTree.
    """
    root = ET.ElementTree()

    rom = ET.Element('rom')

    romid = ET.Element('romid')
    romid.append(ET.Element('xmlid'))
    #Must not be empty
    internalidaddress = ET.Element('internalidaddress')
    internalidaddress.text = '0'
    romid.append(internalidaddress)
    romid.append(ET.Element('internalidstring'))
    #Must not be empty
    ecuid = ET.Element('ecuid')
    ecuid.text = '0'
    romid.append(ecuid)
    romid.append(ET.Element('year'))
    romid.append(ET.Element('market'))
    romid.append(ET.Element('make'))
    romid.append(ET.Element('model'))
    romid.append(ET.Element('submodel'))
    romid.append(ET.Element('transmission'))
    romid.append(ET.Element('memmodel'))
    romid.append(ET.Element('flashmethod'))
    romid.append(ET.Element('filesize'))

    rom.append(romid)

    #Create 2D tables
    for i, row in dataFrame.loc[dataFrame['y_len']==0].iterrows():
        rom.append(createTable2DXml(row))

    #Create 3D tables
    for i, row in dataFrame.loc[dataFrame['y_len']!=0].iterrows():
        rom.append(createTable3DXml(row))

    rom.addprevious(ET.Comment(f'generated by ScoobyTables v{VERSION}'))
    root._setroot(rom)
    ET.indent(root, space=" ", level=0)

    return root

def getInputFormatFromFileName(filename):
    """Guess input file format by extension. Returns DataFormat or None."""
    format = None
    if filename.endswith('.xml'):
        format = DataFormat.xml.value
    elif filename.endswith('.csv'):
        format = DataFormat.csv.value
    return format

HEADER = ['storageaddress','x_len','x_min','x_max','y_len','y_min','y_max','data_type','multiplier','offset','name']
MODEL_DUMP_FILE = 'scoobytables.dmp'
TEST_SAMPLE_SIZE = 0.1
KNN_MIN_2D_RELIABLE_METRIC = 0.5
KNN_MIN_3D_RELIABLE_METRIC = 5
#Predicted output files
PRE_XML_FILENAME = 'output.xml'
PRE_TXT_FILENAME = 'output.txt'
UINT8  = 'UInt8'
UINT16 = 'UInt16'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parserGroupMain = parser.add_mutually_exclusive_group(required=True)
parserGroupMain.add_argument('--train',
                             metavar='<dirname>',
                             help='Train, test and dump model. Specify --test-accuracy to test accuracy.')
parserGroupMain.add_argument('--predict',
                             metavar='<filename.xml>',
                             help='Predict. Get data from <filename.xml> and <filename.bin>')
parserGroupOptions = parser.add_argument('-i', '--input-format',
                                         choices=['xml','csv'],
                                         help='Input data format',
                                         #Detect presence of this argument
                                         action=ArgSetNonDefaultAttr,
                                         default='xml')
parserGroupOptions = parser.add_argument('--model-dump-file',
                                         metavar='<filename>',
                                         help=f'Model dump file name.',
                                         default=f'{MODEL_DUMP_FILE}')
parserGroupOptions = parser.add_argument('-v', '--verbose',
                                         action='store_true',
                                         help='Be verbose.')
parserGroupOptions = parser.add_argument('--test-accuracy',
                                         action='store_true',
                                         help='Test model accuracy during model training.')
parserGroupOptions = parser.add_argument('--test-sample-size',
                                         metavar='<number from 0.0 to 1.0>',
                                         help=f'Test sample size.',
                                         type=float, default=TEST_SAMPLE_SIZE)
parserGroupOptions = parser.add_argument('--dry-run',
                                         action='store_true',
                                         help='Do not save anything to files.')
parserGroupOptions = parser.add_argument('--knn-min-2d-reliable-metric',
                                         metavar='<float number>',
                                         help=f'Minimum reliable metric for 2D tables.',
                                         type=float, default=KNN_MIN_2D_RELIABLE_METRIC)
parserGroupOptions = parser.add_argument('--knn-min-3d-reliable-metric',
                                         metavar='<float number>',
                                         help=f'Minimum reliable metric for 3D tables.',
                                         type=float, default=KNN_MIN_3D_RELIABLE_METRIC)
parserGroupOptions = parser.add_argument('--pre-xml-filename',
                                         metavar='<filename>',
                                         help=f'Predicted XML definitions file name.',
                                         default=f'{PRE_XML_FILENAME}')
parserGroupOptions = parser.add_argument('--dump-txt',
                                         action='store_true',
                                         help='Write predicted data to text file.')
parserGroupOptions = parser.add_argument('--pre-txt-filename',
                                         metavar='<filename>',
                                         help=f'Predicted dataframe text file name.',
                                         default=f'{PRE_TXT_FILENAME}')
parserGroupOptions = parser.add_argument('--version',
                                         action='version',
                                         help='Print version number.',
                                         version=f'{VERSION}')

args = parser.parse_args()

knn_min_2d_reliable_metric = args.knn_min_2d_reliable_metric
knn_min_3d_reliable_metric = args.knn_min_3d_reliable_metric

#Should we save all files or we are in dry run mode?
shouldSave = not args.dry_run
#Path to dataset dir
dataset = args.train
#Model save file
dumpFile = args.model_dump_file
#Input format by default
inputFormat = args.input_format.lower()

if dataset is not None:
    print_console_verbose(f'Loading dataset from {dataset}.')
    print_console_verbose(f'Format is {inputFormat}')
    if inputFormat == DataFormat.csv:
        #CSV defs, can be exported from ScoobyRom
        dataFrame = loadCsvFromDirNormalized(dataset)
    elif inputFormat == DataFormat.xml:
        #ScoobyRom XML format
        dataFrame = loadXmlFromDirNormalized(dataset)
    else:
        print('Unknown input format. Supported formats are XML and CSV.')
        exit()
    #Clean after normalize to get rid of rows where 'name' is empty
    dataFrame = cleanDataFrameRows(dataFrame)
    print_console_verbose("Training.")
    y = dataFrame['name']
    #Remove non-number column(s)
    dataFrame.drop(['name'], axis=1, inplace=True)
    #Split to train and test if needed
    if args.test_accuracy:
        X_train, X_holdout, y_train, y_holdout = train_test_split(dataFrame.values,
                                                                  y,
                                                                  test_size=args.test_sample_size,
                                                                  random_state=17)
    else:
        X_train = dataFrame
        y_train = y

    #Train
    model = trainKNN(X_train, y_train)
    #Print accuracy if needed
    if args.test_accuracy:
        knn_pred = model.predict(X_holdout)
        print_console_verbose('Model accuracy score:')
        print(accuracy_score(y_holdout, knn_pred))
    else:
        print_console_verbose ('Not running accuracy test due to settings.')

    if shouldSave:
        print_console_verbose(f'Saving model to {dumpFile}')
        dumpModel(model, dumpFile)
    else:
        print_console_verbose('Not saving model due to settings.')

    if args.verbose:
        print("Done.")
elif args.predict is not None:
    inputFormatFromFile = getInputFormatFromFileName(args.predict)

    print_console_verbose('Loading data')
    #Try to autodetect input file format
    if hasattr(args, 'input_format_nondefault'):
        print_console_verbose('File format specified by user')
        if inputFormat != inputFormatFromFile:
            print('Did you correctly specify file format? I\'ll continue anyway.')
    elif inputFormatFromFile is not None:
        print_console_verbose('Guessing input file format from file extension')
        inputFormat = inputFormatFromFile
    else:
        print('Can\'t guess file format, using default' )
    print_console_verbose(f'Format is {inputFormat}')

    model = loadModel(dumpFile)
    if inputFormat == DataFormat.csv:
        #CSV defs, can be exported from ScoobyRom
        csvFileName = args.predict
        #This dataframe is the source of exported XML defs
        rawDataFrame = readRawCsvFile(csvFileName)
        #Convert to hex
        rawDataFrame['axis_x_storageaddress'] = rawDataFrame['axis_x_storageaddress'].apply(int).apply(lambda i: f'0x{i:X}')
        rawDataFrame['axis_y_storageaddress'] = rawDataFrame['axis_y_storageaddress'].apply(int).apply(lambda i: f'0x{i:X}')
        rawDataFrame['axis_z_storageaddress'] = rawDataFrame['axis_z_storageaddress'].apply(int).apply(lambda i: f'0x{i:X}')
        dataFrame = parseOneCsvFile(csvFileName)
    elif inputFormat == DataFormat.xml:
        #ScoobyROM defs to predict .xml file
        xmlFileName = args.predict
        binFileName = re.sub('\.xml$', '.bin', xmlFileName)
        (array, nColumns, header) = parseOneXmlFile(xmlFileName, binFileName)
        dataFrame = createDataFrame(array, nColumns, header)
    else:
        print(f'Unknown input format {inputFormat}. Supported formats are XML and CSV.')
        exit()
    tempDataFrame = normalizeDataFrame(dataFrame.copy())
    #Remove non-number column(s)
    tempDataFrame.drop(['name'], axis=1, inplace=True)

    print_console_verbose('Predicting')

    #Classify and write name to array
    predicted_names=model.predict(tempDataFrame)
    #Get distance from current point to nearest neighbors
    distanceArray, _ = model.kneighbors(tempDataFrame)
    #Calculate minimum distance, because there can be "lonely" tables
    #and put it to array
    knn_min_distance = np.min(distanceArray, axis=1)
    
    #Keep all defs data in case of CSV format
    if inputFormat == DataFormat.csv:
        dataFrame = rawDataFrame
    #Overwrite old names with predicted ones
    dataFrame['name'] = predicted_names
    dataFrame['name'] = dataFrame['name'].str.title()
    #Convert storageaddress to hex string
    dataFrame['storageaddress'] = dataFrame['storageaddress'].apply(int).apply(lambda i: f'0x{i:X}')
    dataFrame['knn_min_distance'] = knn_min_distance
    dataFrame['x_len'] = dataFrame['x_len'].apply(int)
    dataFrame['y_len'] = dataFrame['y_len'].apply(int)

    print_console_verbose('Filtering by metric:')
    print_console_verbose(f'    knn_min_2d_reliable_metric = {knn_min_2d_reliable_metric}')
    print_console_verbose(f'    knn_min_3d_reliable_metric = {knn_min_3d_reliable_metric}')

    #Filter by distance
    dataFrame = dataFrame.query(f'(y_len == 0 and knn_min_distance < {knn_min_2d_reliable_metric}) or (y_len > 0 & knn_min_distance < {knn_min_3d_reliable_metric})')
    #Recreate index
    dataFrame = dataFrame.reset_index(drop=True)
    #print(dataFrame.to_string())
    if shouldSave:
        pre_xml_filename = args.pre_xml_filename
        print_console_verbose(f'Saving XML defs to {pre_xml_filename}')
        if inputFormat == DataFormat.csv:
            newXMLtree = createXmlFromCsv(dataFrame.copy())
        elif inputFormat == DataFormat.xml:
            newXMLtree = getCorrectedXMLtree(xmlFileName, dataFrame)
        else:
            print(f'Unknown data format {inputFormat}, cannot convert to XML.')
            exit()
        newXMLtree.write(pre_xml_filename, pretty_print=True, encoding="utf-8", xml_declaration=True)
    else:
        print_console_verbose('Not saving XML defs due to settings.')
    
    if args.dump_txt:
        if shouldSave:
            pre_txt_filename = args.pre_txt_filename
            print_console_verbose(f'Saving text data defs to {pre_txt_filename}')
            dumpedDataFrame = dataFrame.copy()
            #Drop unneeded columns
            dumpedDataFrame.drop(['x_min','x_max','y_min','y_max'], axis=1, inplace=True)
            print(dumpedDataFrame.to_string(), file=open(pre_txt_filename, 'w'))
        else:
            print_console_verbose('Not saving text data defs due to settings.')
    
    print_console_verbose('Done.')
else:
    parser.print_help()
