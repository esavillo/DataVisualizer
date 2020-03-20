# Evan Savillo
# 04/11/16


import csv
import xlrd # Download this
import numpy as np


class Data:
    def __init__(self, filename=None):
        self.filename = filename

        self.raw_headers = []
        self.raw_types = []
        self.raw_data = []
        self.header2raw = {}

        self.matrix_data = np.matrix([], dtype=np.float64)  # matrix of numeric data
        self.header2matrix = {}  # dictionary mapping header string to index of column
        #                                                                 in matrix data
        self.enum2num = {}  # dict for enum to numeric value conversions

        if self.hasFile():
            self.read()


    # Stores raw headers and types in lists, data in a list of lists, and maps
    # headers to column indexes
    def read(self):
        suffix = self.filename[self.filename.rfind('.'):]

        file = open(self.filename, 'rU')

        # For .csv files
        if suffix == '.csv':
            bookworm = csv.reader(file, skipinitialspace=True)

            # Skips past comments
            line = bookworm.next()
            while line[0][0] == '#':
                line = bookworm.next()

            # Fills headers and types lists
            self.raw_headers = line
            self.raw_types = bookworm.next()

            # Fills raw data list of lists
            for row in bookworm:
                self.raw_data += [row]

            # Maps each header to its appropriate column index
            for i in range(len(self.raw_headers)):
                self.header2raw[self.raw_headers[i]] = i

        # For .xls and .xlsx files
        elif suffix == '.xls' or '.xlsx':
            book = xlrd.open_workbook(self.filename)
            sheet = book.sheet_by_index(0)

            i = 0
            for row in sheet._cell_values:
                if str(row[0])[0] != '#':
                    if i > 1:
                        rawrow = []
                        for item in row:
                            rawrow += [str(item)]
                        self.raw_data += [rawrow]
                    elif i == 0:
                        self.raw_headers = row
                        i+=1
                    elif i == 1:
                        self.raw_types = row
                        i+=1

            # Maps each header to its appropriate column index
            for i in range(len(self.raw_headers)):
                self.header2raw[self.raw_headers[i]] = i

        file.close()
        self.scan2Matrix()

    # Handles columns of numeric type data, storing converted data in a numpy matrix
    # also handles columns of data able to be converted to numeric type
    def scan2Matrix(self):
        cols = []
        for header in self.raw_headers:
            i = self.header2raw[header]

            if self.raw_types[i] == 'numeric':
                col = []
                for row in self.raw_data:
                    col += [row[i]]
                cols += [col]
                self.header2matrix[header] = len(cols) - 1

            if self.raw_types[i] == 'percent':
                col = []
                for row in self.raw_data:
                    col += [row[i].rstrip('%')]
                cols += [col]
                self.header2matrix[header] = len(cols) - 1

            if self.raw_types[i] == 'enum':
                col = []
                e = 0
                for row in self.raw_data:
                    if not self.enum2num.has_key(row[i]):
                        self.enum2num[row[i]] = e
                        e += 1

                    col += [self.enum2num[row[i]]]
                cols += [col]
                self.header2matrix[header] = len(cols) - 1

    # Makes the matrix out of cols[[]]; dtype makes numpy handle the actual converting
        self.matrix_data = np.matrix(cols, dtype=np.float64)

    def get_raw_headers(self):
        return self.raw_headers

    def get_raw_types(self):
        return self.raw_types

    def get_raw_num_columns(self):
        return len(self.raw_data[0])

    def get_raw_num_rows(self):
        return len(self.raw_data)

    # Returns a row of data (the type is list) given a row index (int)
    def get_raw_row(self, index):
        return self.raw_data[index]

    # Returns a column of data given a header string
    def get_raw_column(self, header):
        index = self.header2raw[header]

        col = []
        for row in self.raw_data:
            col += [ row[index] ]

        return col

    # Takes a row index (an int) and column header (a string) and returns
    # the raw data (a string) at that location
    def get_raw_value(self, index, header):
        return self.raw_data[index][self.header2raw.get(header)]

    # Returns a list of headers of columns with numeric data
    def get_headers(self):
        headers = range( len(self.header2matrix) )

        for header in self.header2matrix.keys():
            headers[self.header2matrix[header]] = header

        return headers

    # Returns the number of columns of numeric data
    def get_num_columns(self):
        return len(self.get_headers())

    # Take a row index and returns a row of numeric data
    def get_row(self, index):
        return self.matrix_data.T[index]

    # Takes a row index (int) and column header (string) and returns the data
    # in the numeric matrix.
    def get_value(self, index, header):
        return self.matrix_data[self.header2matrix[header], index]

    # Take a list of columns headers and return a matrix with the data for all rows but
    # just the specified columns. Allows the caller to specify a specific set of rows.
    def get_data(self, columns=None, rows=None):
        if columns is None:
            columns = self.get_headers()
        if rows is None:
            rows = range(self.get_raw_num_rows())

        cols = []
        for header in columns:
            cols += [self.header2matrix[header]]

        newtrix = self.matrix_data.take(cols, 0)
        newtrix = newtrix.T.take(rows, 0)

        return newtrix.T

    def get_enum2num(self):
        return self.enum2num.keys()

    # Adds a column of data to the object
    # entries should be a list; header and dtype a string
    def addColumn(self, header, datatype, entries):
        if self.get_raw_num_rows() > len(entries):
            additional = self.get_raw_num_rows()-len(entries)
            for i in range(additional):
                entries+=[entries[-1]]

        self.raw_headers += [header]
        self.raw_types += [datatype]
        self.header2raw[header] = len(self.raw_headers) - 1

        i = 0
        for row in self.raw_data:
            row += [str(entries[i])]
            i += 1


        # Just regenerate all the matrix stuff with updated raw_data
        # (if we need to)
        if datatype == 'numeric' or 'enum' or 'percent':
            self.scan2Matrix()

    def hasFile(self):
        if self.filename is None:
            return False
        else:
            return True

    def giveFile(self, filename):
        if self.hasFile():
            raise IOError('Already has a file')
        else:
            self.filename = file
            self.read()

    # The function should take in a filename and an optional list of the headers
    # of the columns to write to the file.
    def write(self, filename, headers=None):
        if headers is None:
            headers = self.get_headers()

        file = open(filename, 'w+')

        write_string = ''

        # headers
        for header in headers:
            write_string += header + ', '
        write_string = write_string.rstrip(', ') + '\n'

        # types
        for header in headers:
            write_string += 'numeric, '
        write_string = write_string.rstrip(', ') + '\n'

        # data
        data = self.get_data(headers).T
        for row in data:
            write_string += str(row.tolist()).strip('[]') + '\n'

        file.write(write_string)
        file.close()

    def print_string(self):
        print_string = ""
        p = ""
        r = []  # Flags optimal places
        for header in self.get_raw_headers():
            p += "[" + header + "] "
            l = lil(self.get_raw_column(header))# ensuring long data entries adjust
            l = l-len(header)#                                   the header spacing
            p += (" "*l)
            r += [len(p)]

        print_string += p
        print_string += '\n'

        for row in range(self.get_raw_num_rows()):
            p = ""
            for col in range(self.get_raw_num_columns()):
                p += "[" + self.get_raw_value(row, self.get_raw_headers()[col]) + "]"
                p += (" " * (r[col] - len(p)))
            print_string += p
            print_string += '\n'

        return print_string

    # Prints out Data like none other
    def printData(self):
        print self.print_string()

# Returns the length of the longest string in the list
def lil(list):
    longest = 0
    for string in list:
        longest = max(longest, len(string))
    return longest

# Generates a column of data all of a single value
def generateColumn(length, entry):
    entries = [entry]
    for i in range(length-1):
        entries+=[entries[-1]]
    return np.matrix(entries)

class PCAData(Data):
    def __init__(self, ogheaders=[],
                 projd_data=np.matrix([[]]),
                 eva=np.matrix([]),
                 eve=np.matrix([]),
                 mdvals=np.matrix([]),):

        Data.__init__(self, None)

        # New Stuff
        self.ogheaders = ogheaders # subset of original data columns
        self.projd_data = projd_data # numpy matrix
        self.eva = eva # numpy matrix of eigenvalues
        self.eve = eve # numpy matrix of eigenvectors
        self.mdvals = mdvals # numpy matrix of mean data values

        # Old Stuff
        for h in range(len(ogheaders)):
            if h < 10:
                self.raw_headers += ['P0'+str(h)]
            else:
                self.raw_headers += ['P'+str(h)]

            self.raw_types += ['numeric']
            self.header2raw[self.raw_headers[h]] = h

        self.raw_data = self.projd_data.tolist()

        self.scan2Matrix()

    # returns a copy of the eigenvalues as a single-row numpy matrix.
    def get_eigenvalues(self):
        return self.eva.copy()

    # returns a copy of the eigenvectors as a numpy matrix with the eigenvectors as rows.
    def get_eigenvectors(self):
        return self.eve.copy()

    # returns the means for each column in the original data as a single row numpy matrix.
    def get_data_means(self):
        return self.mdvals

    # returns a copy of the list of the headers from the original data used to generate the projected data.
    def get_data_headers(self):
        return self.ogheaders[:]


if __name__ == "__main__":
    file = "data.csv"