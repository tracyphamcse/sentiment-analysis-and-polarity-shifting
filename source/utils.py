from config import ALL_DOMAINS

def display_general_help(file_name):
    print 'Usage: python {}.py [<domain>]'.format(file_name)
    print '\tpython {}.py all\t: to run all domains\n'.format(file_name)
    print 'Supported domains: ' + ', '.join(ALL_DOMAINS) + '\n'
    print 'If no argument, print help'

def display_general_domain_not_supported():
    print 'Domain not found'
    print 'Supported domains: ' + ', '.join(ALL_DOMAINS)

class StringUtils(object):
    def __init__(self, arg):
        super(StringUtils, self).__init__()
        self.arg = arg

    @staticmethod
    def is_empty(string):
        string = string.strip(' \t\n\r')
        return string is None or string == ''

    @staticmethod
    def is_not_empty(string):
        return not StringUtils.is_empty(string)
