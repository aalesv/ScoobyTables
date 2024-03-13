# ScoobyTables

This software is designed for automatic Subaru Denso ROM table markup using machine learning. Currently, only the k-nearest neighbors algorithm is supported because it has distance metrics that can effectively eliminate false positives.

### What's new

#### Version 2024.0311

Improved generated ScoobyRom XML definition file compatibility with ScoobyTables. Now table names are capitalized. Added script that adds suffixes `_A`, `_B` etc or `_1`, `_2` etc to same table names. It could be used after predicted definitions cleanup.

#### Version 2024.0302

Implemented support of CSV definitions format

Added definitions file format autodetect (in predict mode)

#### Version 2024.0228

Implemented support of ScoobyRom XML definitions format

## What ROMs are supported

For now, only Subaru Denso ROMs (and maybe other ROMs that have the same table format) are supported. It includes SH7055, SH7058, and SH72531.

## <a name="use"></a> How to use, briefly

ScoobyTables works with defs in ScoobyRom XML or CSV format. You can [get ScoobyRom here](https://github.com/aalesv/ScoobyRom). This is a fork of ScoobyRom that has support for CSV export and some other improvements.

* Get [ScoobyRom](https://github.com/aalesv/ScoobyRom) if you don't have one
* Download `ScoobyTables.py` to a separate directory
* Install `python` and then `pandas`, `numpy`, `pyarrow`, `scikit-learn` and whatever else is needed
* [Train your model](#train) or [get pre-trained here](https://github.com/aalesv/ScoobyTables-pretrained)
* Copy your ROM to the directory where `ScoobyTables.py` is located
* Open it with ScoobyRom, select all 2D and all 3D tables, and save them by pressing `Ctrl+S` or by clicking the menu `File\Save XML`. Defs will be saved to file `YOUR-ROM-NAME.xml` located in the same directory.
* Run `ScoobyTables.py --predict YOUR-ROM-NAME.xml -v`
* Predicted XML defs should be in the `output.xml` file
* Backup `YOUR-ROM-NAME.xml` if you want and rename `output.xml` to `YOUR-ROM-NAME.xml`
* Reopen your ROM with `ScoobyRom`
* Inspect tables; you should see tables that ScoobyTables has found.
* The software is not 100% accurate. There will be incorrectly detected or named tables, some tables could be missing. Please verify every table before proceeding further!

## <a name="works"></a> How it works and why exactly this way and not otherwise

ScoobyTables uses k-nearest neighbors algorithm. This means that every table should be presented as a point in n-dimensional space. Then the distance from each new, unknown, table to every known table is calculated. Based on this distance, a decision is made about class membership.

The idea is that tables placement does not change significantly from ROM to ROM. Of course, I'm talking about close enough ROMs - same car models, same CPU models, close years. The best illustration of this is Subaru Forester Gen 4 with SH72531 CPU. All the tables in those ROMs are located very similarly relative to each other. That's why this approach works.

The table contents themselves do not matter, only the data that describes the table:

* Table structure relative ROM placement
* Length of table axes
* Data type of X and Y axes - RPM, temperature, engine load, etc.
* Data type of table data itself
* Table multiplier and offset, which are needed to convert from integer to float

Table structure relative ROM placement is easy to calculate if its address is known. Only ScoobyRom stores them. Length of table axes is defined in most defs except ScoobyRom. Data type of X or Y axes could be approximately represented by average number of the axis value - they can be calculated. Fortunately, ScoobyRom stores min and max values in defs' comments. Data type of table data itself are defined in every software. Table multiplier and offset are not stored directly in any definition.

So, no existing defs contain all information. More or less ScoobyRom XML defs are suited, but some info needs to be exctracted from binary ROM file. Or I can modify ScoobyRom to export all the data I need in some format, for example, in CSV, that can be easily imported by `pandas`. Well, I did both.

That's why if you want to use the XML format, you need both `.bin` and `.xml` files. But you can use modified [ScoobyRom](https://github.com/aalesv/ScoobyRom) to export to CSV, and then you don't need a `.bin` file. By the way, I highly recommend to use modified [ScoobyRom](https://github.com/aalesv/ScoobyRom) because it saves all annotated tables and all selected tables, which original ScoobyRom does not do - original saves only annotated tables. And to calculate relative table position, we need to have all the tables in def.

And now a couple of words about results. KNN predicts very well, but it makes many wrong predictions on incomplete data. Hardly someone defined all tables in ROM. So many wrong predictions are made. To filter them, distance thresholds are used - one for 2D tables (because they are more similar) and another for 3D tables. See [CLI help](#cli)

## <a name="train"></a> How to train

First, you need to prepare the data. It is crucial that the same tables in all ROMs be called the same. Symbol case does not matter; a table name may end with `_1`, `_2` etc. or `_A`, `_B` etc. - all this ending will be stripped. For example,  names `Base_Timing_1` and `Base_Timing_A` are good. And names `BaseTimingA` and `Base_Timing1` are not. This greatly affects the accuracy of the prediction.

I assume that you use modified ScoobyRom 0.8.5 or later

* Open the ROM and select all 3D and 2D tables
* Check that there are no erroneous tables in the list at the start and at the end of lists. Unselect erroneous tables if they exist
* Save XML or export to CSV
* Put your CSV files or ROM+XML files in a separate directory, for example, directory `dataset`
* Repeat for all your training ROMs
* Run `ScoobyTables.py --train dataset -v --test-accuracy -i <xml|csv>`
* This means - load all data files from directory `dataset`, be verbose, and test accuracy
* If you use XML, you can omit `-i` parameter. Otherwise, you should specify `-i csv`

## <a name="cli"></a> Command line parameters

One of the arguments `--train` or `--predict` is required. Run with `--help` to see help.

If you specify `--predict`, file format is guessed from the extension. Only ScoobyRom XML and CSV formats are supported.

You can set a distance threshold above which classifying will be ignored: `--knn-min-2d-reliable-metric` for 2D tables, `--knn-min-3d-reliable-metric` for 3D tables.

Happy hacking!
