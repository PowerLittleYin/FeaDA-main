import random
import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

import config
from train.TVA_train import TVA_test_fusion,TVA_train_fusion
from train.Atrain import Atrain,Atest
from train.Vtrain import Vtrain,Vtest
from utils import Metrics
from data_loader import MOSEIDataloader


def main():
    batch_size=config.MOSEI.downStream.batch_size
    print('加载数据')
    train_data = MOSEIDataloader('train', batch_size=batch_size, use_similarity=True, simi_return_mono=False,
                                 use_sampler=False)
    print('train ok')
    valid_data = MOSEIDataloader('valid',shuffle=False, num_workers=0,batch_size=batch_size)
    print('valid ok')
    test_data = MOSEIDataloader('test', shuffle=False, num_workers=0,batch_size=batch_size)
    print('test ok')
    metrics = Metrics()
    
    config.seed = 430
    # Vtrain(config, metrics, config.seed, train_data, valid_data)
    # Vtest(config, metrics, test_data)

    # Atrain(config, metrics, config.seed, train_data, valid_data)
    # Atest(config, metrics, test_data)

    TVA_train_fusion(config, metrics, config.seed, train_data, valid_data)
    TVA_test_fusion(config, metrics,  test_data, )


if __name__ == '__main__':
    main()
