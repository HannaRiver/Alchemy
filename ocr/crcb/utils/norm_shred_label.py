from lca2ca import cncurrency
from ca2lca import ca2lca

def normShredLabel(label, rec_item, rec):
    '''
    对银行标签进行归一化转换
    '''
    if rec_item == 'CA':
        norm = cncurrency(label) if ca2lca(rec) != label else rec
        return norm
    else:
        pass
    return label