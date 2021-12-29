import utills.spkit_ex as ut
import os

if __name__ == '__main__':
    """result에 저장"""
    foler = "12dh"
    if not os.path.isdir(f'result/{foler}'):
        os.mkdir(f'result/{foler}')
    # ut.tuning_beta_plot(data_folder, "elim")
    # ut.tuning_beta(data_folder, "elim") #soft, linAtten, elim
    ut.atar_plotting(foler, plotting=False, cut=True) #plotting=false
    