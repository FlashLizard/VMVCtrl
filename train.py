from tools.train.train_ctrl_t2v_enterance import train_ctrl_t2v_entrance
from utils.config import Config

if __name__ == '__main__':
    cfg_update = Config(load=True)
    print(cfg_update.cfg_dict)
    train_ctrl_t2v_entrance(cfg_update.cfg_dict)
