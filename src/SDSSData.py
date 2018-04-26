import numpy as np

class SDSSData:
    ''' A class for managing photometric and spectroscopic data '''
    def __init__(self, spec_csv, phot_csv, header_line=1, delimiter=','):
        # Load headers
        self.spec_h = {}
        self.phot_h = {}
        with open(spec_csv) as filep:
            for _ in range(header_line-1):
                next(filep)
            header = filep.readline().strip().split(',')
            for i in range(len(header)):
                self.spec_h[header[i]] = i
        with open(phot_csv) as filep:
            for _ in range(header_line-1):
                next(filep)
            header = filep.readline().strip().split(',')
            for i in range(len(header)):
                self.phot_h[header[i]] = i
        # Load data
        self.specd = np.genfromtxt(spec_csv, skip_header=header_line, delimiter=delimiter)
        self.photd = np.genfromtxt(phot_csv, skip_header=header_line, delimiter=delimiter)
        self.views = {}
    def spec(self, keys=None):
        ''' Get spec data with or without a key '''
        if not keys:
            return self.specd
        if type(keys) == type([]):
            a = []
            for key in keys:
                a.append(self.spec_h[key])
            return self.specd[:, a]
        return self.specd[:, self.spec_h[keys]]

    def phot(self, keys=None):
        ''' Get phot data with or without a key '''
        if not keys:
            return self.photd
        if type(keys) == type([]):
            a = []
            for key in keys:
                a.append(self.phot_h[key])
            return self.photd[:, a]
        return self.photd[:, self.phot_h[keys]]
    def add_view(self, name, view):
        ''' Submit a view '''
        self.views[name] = view
    def view(self, name):
        ''' Access a stored view '''
        return self.views[name]

def main():
    data = SDSSData("testdata.csv", "testdata.csv")
    print(data.phot('a'))
    data.add_view("color", data.phot('a') - data.phot('b'))
    print(data.view("color"))




if __name__ == "__main__":
    main()
