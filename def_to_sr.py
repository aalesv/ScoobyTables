# IDA 6.8 python script
#
# Converts RomRaider definitions to ScoobyRom
# Outputs to output.xml in the same directory where ROM is
#
# Version 2026.0512
#
import glob
import os
import xml.etree.cElementTree as ET


class DefType:
    """Definitions type"""

    Unknown = 0
    RR = 1
    EF = 2


class StorageType2D:
    """Storage Type for 2D tables"""

    FLOAT = 0
    UINT8 = 0x400
    UINT16 = 0x800


class StorageType3D:
    """Storage Type for 3D tables"""

    FLOAT = 0
    UINT8 = 0x4000000
    UINT16 = 0x8000000


def detectDefType(xmlRoot):
    """Detect XML def file type - RomRaider or EcuFlash"""
    res = DefType.Unknown
    if xmlRoot.tag.lower() == 'roms':
        res = DefType.RR
    elif xmlRoot.tag.lower() == 'rom':
        res = DefType.EF
    return res


def copyXmlAttribute(eFrom, eTo):
    """Copy all attributes, returns new attribute"""
    for e in eFrom.attrib:
        eTo.set(e, eFrom.get(e))
    return eTo


def getRRTables(xmlRoot, calId):
    """Get only tables for RomRaider defs."""
    tables = ET.Element('tables')
    # Find <romid> that contains <xmlid> with needed value and then go one level up
    xpath_rom = './rom/romid/[xmlid="{0}"]/..'.format(calId)
    rom = xmlRoot.find(xpath_rom)
    base = rom.get('base')
    print('Building defs for {0}'.format(calId))
    if base is None:
        # This is the BASE defs. Return everythnig.
        xath_t2d = './table[@type="2D"]'
        t2d = rom.findall(xath_t2d)
        xpath_t3d = './table[@type="3D"]'
        t3d = rom.findall(xpath_t3d)
        for t in t2d + t3d:
            tables.append(t)
    else:
        # Include base tables
        tables_base = getRRTables(xmlRoot, base)
        xmlThis = rom
        # Overwrite all existing tables with new
        # Only top and first level
        for t_base in tables_base:
            # search in current filtered <rom> tag
            xb = './table[@name="{0}"]'.format(t_base.get('name'))
            t_this = xmlThis.find(xb)
            # Overwrite every element with same name in base XML
            # with new attributes in this XML
            if t_this is not None:
                t = copyXmlAttribute(t_this, t_base)
                # Copy X and Y axis table attributes
                for t_b in t_base:
                    if t_b.get('type') is None:
                        continue
                    xb1 = './table[@type="{0}"]'.format(t_b.get('type'))
                    t1 = t_this.find(xb1)
                    if t1 is not None:
                        # I'm not sure why it's working like this...
                        copyXmlAttribute(t1, t_b)
                        # ... and not like this
                        # e=copyXmlAttribute(t1, t_b)
                        # t.append(e)
                # End Copy X and Y axis table attributes
                tables.append(t)
            else:
                tables.append(t_base)
        # Add nonexistent tables
        xpath_this_table = './table'
        tables_this = rom.findall(xpath_this_table)
        for t_this in tables_this:
            xb = './table[@name="{0}"]'.format(t_this.get('name'))
            t_base = tables_base.find(xb)
            if t_base is None:
                tables.append(t_this)
    print('Done building defs for {0}'.format(calId))
    return tables


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


def prettyHex(hex):
    """Add 0x to hex number if needed"""
    h = '0x0'
    if hex is not None and not hex.startswith('0x'):
        h = '0x{0}'.format(hex)
    return h


def MakeDwordSafe(address):
    # Check if it is tail byte
    if isTail(GetFlags(address)):
        # Undefine it
        MakeUnknown(address, 4, DOUNK_SIMPLE)
    return MakeDword(address)


def findFirstDataXrefTo(address):
    """Find first data reference to address"""
    a = DfirstB(address)
    if a == BADADDR:
        # Convert address to hex string
        # Find reference
        a = FindBinary(0, 1, hex(address))
        MakeDwordSafe(a)
    return a


def getTable2DStructStart(valuesAddr):
    """Find address for table structure. valueAddr must be address of 2D table values"""
    a = findFirstDataXrefTo(valuesAddr)
    a = a - 8
    return a


def getTable3DStructStart(valuesAddr):
    """Find address for table structure. valueAddr must be address of 3D table values"""
    a = findFirstDataXrefTo(valuesAddr)
    a = a - 12
    return a


def isAddressValid(address):
    """If address don't fit in RAM, return True, otherwise False"""
    res = False
    if address < 0xFFFF0000:
        res = True
    return res


def getTable2DStorageType(tableStructAddress):
    """Returns Table 2D storage type taken from ROM"""
    r = ''
    s = Word(tableStructAddress + 2)
    if s == StorageType2D.FLOAT:
        r = 'Float'
    elif s == StorageType2D.UINT8:
        r = 'UInt8'
    elif s == StorageType2D.UINT16:
        r = 'UInt16'
    return r


def getTable3DStorageType(tableStructAddress):
    """Returns Table 3D storage type taken from ROM"""
    r = ''
    s = Dword(tableStructAddress + 16)
    if s == StorageType3D.FLOAT:
        r = 'Float'
    elif s == StorageType3D.UINT8:
        r = 'UInt8'
    elif s == StorageType3D.UINT16:
        r = 'UInt16'
    return r


def transofrmRRToSR(xmlRoot):
    """Transform RomRaider defs to ScoobyRom defs"""
    rom = ET.Element('rom')
    romid = xmlRoot.find('./romid')
    rom.append(romid)
    for t in xmlRoot.findall('./table[@type="2D"]'):
        # table2D
        addr = t.get('storageaddress')
        addr_16 = int(addr, 16)
        addr_struc = int(getTable2DStructStart(addr_16))
        if not isAddressValid(addr_struc):
            continue
        storageaddress = hex(addr_struc)
        attributesArray = [
            ('category', t.get('category')),
            ('name', t.get('name')),
            ('storageaddress', storageaddress),
        ]
        table2d = createElement('table2D', attributesArray)
        # axisX
        y_axis = t.find('./table[@type="Y Axis"][@type="X Axis"]')
        if y_axis is not None:
            y_axis_scaling = y_axis.find('./scaling')
            if y_axis.get('storageaddress') is None:
                print(
                    'transofrmRRToSR: {1}: storageaddress is None at {0}'.format(
                        ET.tostring(y_axis), romid.find('./xmlid').text
                    )
                )
            attributesArray = [
                ('storageaddress', prettyHex(y_axis.get('storageaddress'))),
                ('name', y_axis.get('name')),
                (
                    'unit',
                    y_axis_scaling.get('units') if y_axis_scaling is not None else '',
                ),
            ]
            axisx = createElement('axisX', attributesArray)
            table2d.append(axisx)
        # values
        values_storageaddress = t.get('storageaddress')
        scaling = t.find('./scaling')
        storagetype = t.get('storagetype')
        attributesArray = [
            ('storageaddress', prettyHex(values_storageaddress)),
            ('unit', scaling.get('units') if scaling is not None else ''),
            (
                'storagetype',
                storagetype
                if storagetype is not None
                else getTable2DStorageType(int(storageaddress, 16)),
            ),
        ]
        values = createElement('values', attributesArray)
        table2d.append(values)
        rom.append(table2d)
    for t in xmlRoot.findall('./table[@type="3D"]'):
        # table3D
        addr = t.get('storageaddress')
        addr_16 = int(addr, 16)
        addr_struc = int(getTable3DStructStart(addr_16))
        if not isAddressValid(addr_struc):
            continue
        storageaddress = hex(addr_struc)
        attributesArray = [
            ('category', t.get('category')),
            ('name', t.get('name')),
            ('storageaddress', storageaddress),
        ]
        table3d = createElement('table3D', attributesArray)
        # axisX
        x_axis = t.find('./table[@type="X Axis"]')
        if x_axis is not None:
            x_axis_scaling = x_axis.find('./scaling')
            attributesArray = [
                ('storageaddress', prettyHex(x_axis.get('storageaddress'))),
                ('name', x_axis.get('name')),
                (
                    'unit',
                    x_axis_scaling.get('units') if x_axis_scaling is not None else '',
                ),
            ]
            axisx = createElement('axisX', attributesArray)
            table3d.append(axisx)
        # axisY
        y_axis = t.find('./table[@type="Y Axis"]')
        if y_axis is not None:
            y_axis_scaling = y_axis.find('./scaling')
            attributesArray = [
                ('storageaddress', prettyHex(y_axis.get('storageaddress'))),
                ('name', y_axis.get('name')),
                (
                    'unit',
                    y_axis_scaling.get('units') if y_axis_scaling is not None else '',
                ),
            ]
            axisx = createElement('axisY', attributesArray)
            table3d.append(axisx)
        # values
        values_storageaddress = t.get('storageaddress')
        scaling = t.find('./scaling')
        storagetype = t.get('storagetype')
        attributesArray = [
            ('storageaddress', prettyHex(values_storageaddress)),
            ('unit', scaling.get('units') if scaling is not None else ''),
            (
                'storagetype',
                storagetype
                if storagetype is not None
                else getTable3DStorageType(int(storageaddress, 16)),
            ),
        ]
        values = createElement('values', attributesArray)
        table3d.append(values)
        rom.append(table3d)
    return rom


def processRR(xmlRoot, calId):
    """Build full XML with includes for RomRaider defs. Return ElementTree."""
    # Find <romid> that contains <xmlid> with needed value and then go one level up
    xpath = './rom/romid/[xmlid="{0}"]/..'.format(calId)
    rom = xmlRoot.find(xpath)
    if rom is None:
        raise Exception('{0} not found'.format(calId))
    fullXml = ET.Element('rom')
    # Copy <romid>
    fullXml.append(rom.find('./romid'))
    tables = getRRTables(xmlRoot, calId)

    t2d_ftm = './table[@type="2D"][@storageaddress]'
    t2d = tables.findall(t2d_ftm)
    t3d_ftm = './table[@type="3D"][@storageaddress]'
    t3d = tables.findall(t3d_ftm)
    for t in t2d + t3d:
        fullXml.append(t)

    print('Parsing ROM to create ScoobyRom definitions...')
    srXml = transofrmRRToSR(fullXml)
    return srXml


def getDefId(xmlFile):
    """Get definition ID from XML file. File must be in EcuFlash format"""
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    xmlid = root.find('./romid/[xmlid]/xmlid')
    return xmlid.text if xmlid is not None else None


def createRrLikeFromEF(calId, defsFilesDict, xmlRoot=None):
    """
    Take definition file, append all includes
    and transform to RomRaider-like format -
    put all <rom> tags inside root <roms> tag.
    Add RomRaider-specific attributes.
    """
    if xmlRoot is None:
        xmlRoot = ET.Element('roms')
    try:
        def_file = defsFilesDict[calId]
    except KeyError:
        print('File for {0} not found.'.format(calId))
        def_file = AskFile(0, '*.xml', 'Select definitions file for {0}'.format(calId))
    tree = ET.parse(def_file)
    root = tree.getroot()
    for e in root.findall('.//*[@address]'):
        a = e.get('address')
        if a is not None:
            e.set('storageaddress', a)
    include = root.find('./include')
    if include is not None:
        rom = root.set('base', include.text)
        createRrLikeFromEF(include.text, defsFilesDict, xmlRoot)
    xmlRoot.append(root)
    return xmlRoot


def removeNoneAttrib(elem):
    """Remove None attributes from XML tree"""
    elem.attrib = {k: elem.attrib[k] for k in elem.attrib if elem.attrib[k] is not None}
    for subelem in elem:
        removeNoneAttrib(subelem)


def processEF(xmlFile, calId):
    dir = os.path.dirname(os.path.abspath(xmlFile))
    def_files = glob.glob(os.path.join(dir, '*.xml'))
    print('Gathering defs info, please wait...')
    available_defs = {getDefId(f): f for f in def_files}
    rr_like_xml = createRrLikeFromEF(calId, available_defs)
    fullXml = processRR(rr_like_xml, calId)
    # removeNoneAttrib(fullXml)
    tree = ET.ElementTree()
    tree._setroot(rr_like_xml)
    tree.write('output-debug.xml', encoding='utf-8', xml_declaration=True)
    return fullXml


def processCalId(xmlFile, calId):
    tree = ET.parse(xmlFile)
    xmlRoot = tree.getroot()
    defType = detectDefType(xmlRoot)
    if defType == DefType.RR:
        fullXml = processRR(xmlRoot, calId)
    elif defType == DefType.EF:
        fullXml = processEF(xmlFile, calId)
    else:
        raise Exception('Unknown definitions format')
    return fullXml


def guessCalId(xmlFile):
    """Guess what CAL Id has this ROM"""
    tree = ET.parse(xmlFile)
    xmlRoot = tree.getroot()
    defType = detectDefType(xmlRoot)
    # Find all <romid> where <internalidaddress> is present
    if defType == DefType.RR:
        romid_xpath = './rom/romid/[internalidaddress]'
    elif defType == DefType.EF:
        romid_xpath = './romid/[internalidaddress]'
    else:
        return ''
    romIds = xmlRoot.findall(romid_xpath)
    xmlIdDict = {}
    # Create dictionary {CALID:Address}
    for id in romIds:
        idString = id.find('./internalidstring').text
        idAddr = prettyHex(id.find('./internalidaddress').text)
        xmlIdDict.update({idString: idAddr})
    calId = ''
    # Check if there is defs for this ROM
    for id in xmlIdDict:
        addr = xmlIdDict.get(id)
        romId = GetString(int(addr, 16), len(id), ASCSTR_C)
        if romId == id:
            calId = id
    return calId


OUTPUT_FILE_NAME = 'output.xml'
file_path = AskFile(0, '*.xml', 'Select definitions file')
if file_path == 0:
    print('No file is selected, exiting')
else:
    calId = guessCalId(file_path)
    calId = AskStr(calId, 'Enter Cal ID')
    if calId is None:
        print('No Cal ID entered, exiting')
        exit(0)
    print('CAL ID is {0}, defs file is {1}, processing...'.format(calId, file_path))
    r = processCalId(file_path, calId)
    tree = ET.ElementTree()
    tree._setroot(r)
    tree.write(OUTPUT_FILE_NAME, encoding='utf-8', xml_declaration=True)
    print('Done. ScoobyRom XML written to {0}'.format(OUTPUT_FILE_NAME))
