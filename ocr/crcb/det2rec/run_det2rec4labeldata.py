import os.path as osp
from alchemy_config import cfg, cfg_from_file, cfg_from_list
from det2rec4labeldata import main as det2rec
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='DET2REC4LABELDATE ARGS')
    parser.add_argument('--rec_item', default='CA')
    
    return parser.parse_args()

RecItem2Cfg = {
    'CA': cfg.CRCB_CFG_NAME.CA_PRINT,
    'LA': cfg.CRCB_CFG_NAME.LA_PRINT,
    'CD': cfg.CRCB_CFG_NAME.CD_PRINT,
    'LD': cfg.CRCB_CFG_NAME.LD_PRINT,
    'Chn': cfg.CRCB_CFG_NAME.CHN_PRINT,
    'Num': cfg.CRCB_CFG_NAME.NUM_PRINT,
}

if __name__ == '__main__':
    args = parse_args()
    cfg_file = RecItem2Cfg[args.rec_item]
    cfg_from_file(osp.join(cfg.CRCB_DIR, 'det2rec', cfg_file))
    
    det2rec()