from model.pmrid.pmrid_api import PMRID_API
import torch

if __name__=='__main__':

    pmrid_api = PMRID_API(
        200, 
        10,
        0.01,
        'cuda:0',
        './logs/pmrid_l2/',
        './params/pmrid_l2/',
        './data/right_value_list.txt',
        './data/right_value_list.txt',
        True,
        './model/pmrid/pmrid_pretrained.ckp'
        )
    pmrid_api.train_and_value() # train and value
    # pmrid_api.test('./params/pmrid_l2/100.ckp','./output/pmrid_l2_dataset/value/right/') # test
