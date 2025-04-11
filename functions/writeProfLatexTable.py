import numpy as np


class LatexTable:
    def __init__(self, file, sidewaystable=False):
        self.key_sidewaystable = sidewaystable
        self.file = file

    def writeLatexTableHeader(self, caption = None,label = None):

        key_sidewaystable = self.key_sidewaystable
        file = self.file

        if key_sidewaystable:
            file.write('\\begin{sidewaystable}[ht]\n')
        else:
            file.write('\\begin{table}[ht]\n')

        file.write('\\centering\n')
        if not caption == None:
            file.write('\caption{' + caption +'}\n')
        if not label == None:
            file.write('\label{'   + label   +'}\n')

    def writeLatexTableEnd(self):
        
        file=self.file

        if self.key_sidewaystable:
            file.write('\\end{sidewaystable}\n')
            self.key_sidewaystable = False
        else:
            file.write('\\end{table}\n')

    def writeLatexSubTableHeader(self ,caption = None,label = None):
        file=self.file

        file.write('\\begin{subtable}{\\linewidth}\n')
        file.write('\\centering')
        if not caption == None:
            file.write('\\caption{' +  caption +'}\n')
        if not label == None:
            file.write('\label{'   + label   +'}\n')

    def writeLatexSubTableEnd(self):
        file=self.file
        file.write('\\end{subtable}\n')

    def writeLatexTableAddSpace(self):
        file = self.file
        file.write('\\vspace{0.75\\baselineskip}')


# This function writes the data part (tabular) into a Latex table.
# IMPORTANT: data can be a one or two deminsional numpy array.
    def writeProfLatexTabular(self, data, columnsNames=[],rowsNames=[], **kwargs):
        file=self.file
        # default arguments values
        float_format = "%.3f"
        rows_title = ' & '
        multicolumn = None
        multicolumn_lines = None
        midrule_at_lines = None

        # Optional arguments of function.
        # First print them.
        for key, value in kwargs.items():
            print("{0} = {1}".format(key, value))
            if  key == 'float_format':
                float_format = value
            elif key == 'rows_title':
                rows_title = value + rows_title
            elif key == 'multicolumn':
                multicolumn = value
            elif key == 'multicolumn_lines':
                multicolumn_lines = value
            elif key == 'midrule_at_lines':
                midrule_at_lines = value

        if data.ndim == 2:
            nColums = data.shape[1]
            nRows   = data.shape[0]

        if data.ndim == 1:
            nColums = len(data)
            nRows   = 1
            data = data.reshape(1,-1)
            print(data.shape)

        print('nColums= %i, nRows= %i' %(nColums,nRows))

        file.write('\\begin{tabular}{ ')

        if rowsNames:
            file.write('l')

        for i in range(nColums):
            file.write('c')

        file.write(' } \n')
        file.write('\\toprule \n')

        if not (multicolumn is None):
            print(multicolumn)
            file.write(multicolumn + "\n")
            if not (multicolumn_lines is None):
                file.write(multicolumn_lines)


#  writing the columns names
        if columnsNames:
            print('columnsNames: ')
            print(columnsNames)
            assert len(columnsNames)==nColums

            if rowsNames:
                print('rowsNames')
                print(rowsNames)
                print(type(rowsNames))

                file.write(rows_title)

            for i in np.arange(0, len(columnsNames)):
                file.write(columnsNames[i])

                if i == len(columnsNames) - 1:
                    file.write(' \\\\ \n')
                else:
                    file.write(' & ')

            file.write('\\midrule \n')
#  == end ==

#  writing the data block
        for row in range(nRows):
            if midrule_at_lines:
                if row in midrule_at_lines:
                    file.write('\\midrule \n')

            if rowsNames:
                file.write(rowsNames[row] + ' & ')
            for column in range(nColums):
                if np.isnan(data[row, column]):
                    string = '{}'
                else:
                    string = float_format % data[row, column]

                file.write(string)

                if column == nColums-1:
                    file.write(' \\\\ \n')
                else:
                    file.write(' & ')

#  == end ==

        file.write('\\bottomrule \n')
        file.write('\\end{tabular} \n')

        return

#====== End Method writeProfLatexTabular ======

if __name__ == '__main__':

    a=np.eye(2)
    file= open('xxx.tex','w')
    latex_table=LatexTable(file=file,sidewaystable=True)
    latex_table.writeLatexTableHeader(caption=None, label=None)
    latex_table.writeProfLatexTabular(data=a,float_format="%.2f")
    latex_table.writeLatexTableEnd()
    file.close()