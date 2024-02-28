import argparse
import glob
import re
import struct
import xml.etree.ElementTree as ET
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

VERSION = '2024.0228'

def normalizeDataFrame(dataFrame):
    """
    Normalizes DataFrame for one ROM
    """
    #Convert to float
    #Last column is string
    header = dataFrame.columns
    for i in header[:len(header)-1]:
        dataFrame[i] = dataFrame[i].astype(float)

    #Convert to distance from start of data area
    min = dataFrame['distance'].min()
    max = dataFrame['distance'].max() - min
    dataFrame['distance'] -= min
    dataFrame['distance'] /= max
    dataFrame['distance'] *= 100
    dataFrame['x_len'] *= 5
    dataFrame['y_len'] *= 5
    #Increase scale of multiplier to make it more significant
    dataFrame['multiplier'] *= 10000
    #
    dataFrame['data_type'] *= 25
    return dataFrame

def createDataFrame(array, nColumns, header):
    """
    Create dataframe from array 'array'. 
    Dataframe will contain 'nColumns' columns and 'header' header.
    """
    array = array.reshape(-1, nColumns)
    dataFrame = pd.DataFrame(array, columns=header)
    return dataFrame

def createAndNormalizeDataFrame(array, nColumns, header):
    """
    Create and normalize dataframe from array 'array'. 
    Dataframe will contain 'nColumns' columns and 'header' header
    """
    df = createDataFrame(array, nColumns, header)
    df = normalizeDataFrame(df)
    return df

def cleanDataFrame(dataFrame):
    """Delete strings with empty 'name' column values"""
    return dataFrame.query('name != ""')

def loadFromDirNormalized(dirname):
    """
    Load dataset from ROM binary files located in 'dirname'.\n
    XML must be in ScoobyRom format
    """
    files = glob.glob(f'{dirname}/*.xml')
    dataFrame = pd.DataFrame()
    for f in files:
        binFileName = re.sub('\.xml$', '.bin', f)
        (t, nColumns, header) = parseOneFile(f, binFileName)
        tempDataFrame = createAndNormalizeDataFrame(t, nColumns, header)
        frames = [dataFrame, tempDataFrame]
        dataFrame = pd.concat(frames, ignore_index=True)
    return dataFrame

def parseOneFile(xmlFileName, binFileName):
    """
    Parse one ROM file. Returns: (dataset array, number of columns, headers name array of string).\n
    XML must be in ScoobyRom format    """
    tree = ET.parse(xmlFileName)
    root = tree.getroot()
    #Open for read in binary mode
    binFile = open(binFileName, 'rb')
    ds = np.array([])
    for child in root:
        if child.tag == 'table2D' or child.tag == 'table3D':
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
                y_len = 0
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
            #Header. Place name last!
            #storageaddress must be converted to distance later
            h =          ['distance',      'x_len','y_len','data_type','multiplier','offset','name']
            t = np.array([ storageaddress,  x_len,  y_len,  data_type,  multiplier,  offset,  name])
            (ncolumns,) = t.shape
            ds = np.append(ds, t)
    return ds, ncolumns, h

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
    #print (array)
    tree = ET.parse(xmlFileName)
    root = tree.getroot()
    for a in array:
        #Convert to int from hex string
        addr = int(a[0], base=16)
        #Find all tags where 'storageaddress'==addr (in hexadecimal representation)
        t = root.find(f'.//*[@storageaddress="0x{addr:X}"]')
        #Replace 'name' sttribute
        t.set('name', a[1])
    return tree

MODEL_DUMP_FILE = 'scoobytables.dmp'
TEST_SAMPLE_SIZE = 0.1
KNN_MIN_2D_RELIABLE_METRIC = 1
KNN_MIN_3D_RELIABLE_METRIC = 10
PRE_XML_FILENAME = 'output.xml'
PRE_TXT_FILENAME = 'output.txt'

parser = argparse.ArgumentParser()
parserGroupOptions = parser.add_argument('--model-dump-file',
                                         metavar='<filename>',
                                         help=f'Model dump file name. {MODEL_DUMP_FILE} by default.',
                                         default=f'{MODEL_DUMP_FILE}')
parserGroupOptions = parser.add_argument('--version',
                                         action='store_true',
                                         help='Print version number.')
parserGroupOptions = parser.add_argument('-v', '--verbose',
                                         action='store_true',
                                         help='Be verbose. No by default.')
parserGroupOptions = parser.add_argument('--test-accuracy',
                                         action='store_true',
                                         help='Test model accuracy during model training. No by default.')
parserGroupOptions = parser.add_argument('--test-sample-size',
                                         metavar='<number from 0.0 to 1.0>',
                                         help=f'Test sample size. {TEST_SAMPLE_SIZE} by default.',
                                         type=float, default=TEST_SAMPLE_SIZE)
parserGroupOptions = parser.add_argument('--dry-run',
                                         action='store_true',
                                         help='Do not save anything to files. Save by default.')
parserGroupOptions = parser.add_argument('--knn-min-2d-reliable-metric',
                                         metavar='<float number>',
                                         help=f'Minimum reliable metric for 2D tables. {KNN_MIN_2D_RELIABLE_METRIC} by default.',
                                         type=float, default=KNN_MIN_2D_RELIABLE_METRIC)
parserGroupOptions = parser.add_argument('--knn-min-3d-reliable-metric',
                                         metavar='<float number>',
                                         help=f'Minimum reliable metric for 3D tables. {KNN_MIN_3D_RELIABLE_METRIC} by default.',
                                         type=float, default=KNN_MIN_3D_RELIABLE_METRIC)
parserGroupMain = parser.add_mutually_exclusive_group()
parserGroupMain.add_argument('--train',
                             metavar='<dirname>',
                             help='Train, test and dump model. Specify --test-accuracy to test accuracy.')
parserGroupMain.add_argument('--predict',
                             metavar='<filename.xml>',
                             help='Predict. Get data from <filename.xml> and <filename.bin>')
parserGroupOptions = parser.add_argument('--pre-xml-filename',
                                         metavar='<filename>',
                                         help=f'Predicted XML definitions file name. {PRE_XML_FILENAME} by default.',
                                         default=f'{PRE_XML_FILENAME}')
parserGroupOptions = parser.add_argument('--dump-txt',
                                         action='store_true',
                                         help='Write predicted data to text file. Do not write by default.')
parserGroupOptions = parser.add_argument('--pre-txt-filename',
                                         metavar='<filename>',
                                         help=f'Predicted dataframe text file name. {PRE_TXT_FILENAME} by default.',
                                         default=f'{PRE_TXT_FILENAME}')

args = parser.parse_args()

knn_min_2d_reliable_metric = args.knn_min_2d_reliable_metric
knn_min_3d_reliable_metric = args.knn_min_3d_reliable_metric

#Should we save all files or we are in dry run mode?
shouldSave = not args.dry_run
#Path to dataset dir
dataset = args.train
#Model save file
dumpFile = args.model_dump_file

if dataset is not None:
    print_console_verbose(f'Loading dataset from {dataset}.')
    df = loadFromDirNormalized(dataset)
    #Clean after normalize to get rid of rows where 'name' is empty
    df = cleanDataFrame(df)
    print_console_verbose("Training.")
    y = df['name']
    #Remove non-number column(s)
    df.drop(['name'], axis=1, inplace=True)
    #Split to train and split if needed
    if args.test_accuracy:
        X_train, X_holdout, y_train, y_holdout = train_test_split(df.values,
                                                                  y,
                                                                  test_size=args.test_sample_size,
                                                                  random_state=17)
    else:
        X_train = df
        y_train = y

    #Trian
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
    print_console_verbose('Loading data')

    model = loadModel(dumpFile)
    #ScoobyROM defs to predict .xml file
    xmlFileName = args.predict
    binFileName = re.sub('\.xml$', '.bin', xmlFileName)

    (array, nColumns, header) = parseOneFile(xmlFileName, binFileName)
    dataFrame = createDataFrame(array, nColumns, header)
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
    
    #Overwrite old names with predicted ones
    dataFrame['name'] = predicted_names
    #In orig dataframe 'distance' is still 'storageaddress'
    #because no normalization was done
    dataFrame['storageaddress'] = dataFrame['distance'].apply(int).apply(hex)
    dataFrame['knn_min_distance'] = knn_min_distance
    dataFrame['x_len'] = dataFrame['x_len'].apply(int)
    dataFrame['y_len'] = dataFrame['y_len'].apply(int)

    print_console_verbose('Filtering by metric:')
    print_console_verbose(f'    knn_min_2d_reliable_metric = {knn_min_2d_reliable_metric}')
    print_console_verbose(f'    knn_min_3d_reliable_metric = {knn_min_3d_reliable_metric}')

    dataFrame = dataFrame.query(f'(y_len == 0 and knn_min_distance < {knn_min_2d_reliable_metric}) or (y_len > 0 & knn_min_distance < {knn_min_3d_reliable_metric})')
    #Recreate index
    dataFrame = dataFrame.reset_index(drop=True)
    #print(dataFrame.to_string())
    if shouldSave:
        print_console_verbose(f'Saving XML defs to {xmlFileName}')
        newXMLtree = getCorrectedXMLtree(xmlFileName, dataFrame)
        pre_xml_filename = args.pre_xml_filename
        newXMLtree.write(pre_xml_filename)
    else:
        print_console_verbose('Not saving XML defs due to settings.')
    
    if args.dump_txt:
        if shouldSave:
            pre_txt_filename = args.pre_txt_filename
            print_console_verbose(f'Saving text data defs to {pre_txt_filename}')
            dumpedDataFrame = dataFrame.copy()
            dumpedDataFrame.drop(['distance'], axis=1, inplace=True)
            print(dumpedDataFrame.to_string(), file=open(pre_txt_filename, 'w'))
        else:
            print_console_verbose('Not saving text data defs due to settings.')
    
    print_console_verbose('Done.')
elif args.version:
    print(VERSION)
else:
    parser.parse_args('-h')