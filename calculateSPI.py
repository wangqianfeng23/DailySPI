import utils
import numpy as np
import pandas as pd
import glob
import os

# ------------------------------------------------------------------------------
def cal_SPI(
        csvpath:str,
        scale: int,
        type: str,
        outcsv:str,
):
    """
            You can use this function if you want batch calculation

            :param csvpath: The path to the folder of the target data store.
            :param scale: The scale size of the calculation. The data type is an integer. 
                    One month is 30, three months is 3 times 30, six months is 6 times 60, and so on.
            :param type: str, Calculate the monthly or daily SPI. 
            :param outcsv: Path to the folder where the result file is stored
            :return: no return
    """
    #read csv
    data = pd.read_csv(csvpath)
    pre = np.asarray(data['Precip'])

    #calculate SPI
    culpre=utils.scale_values(pre,scale,type)
    f_alpa, f_beta, f_P0=utils.fit_gamma_para(culpre)
    rawSPI=utils.caculate_SPI(culpre,f_alpa, f_beta, f_P0)
    SPI = np.clip(rawSPI, -3, 3).flatten()

    #save SPI
    np.savetxt(outcsv, SPI)

# ------------------------------------------------------------------------------
def batch_cal_SPI(
        filepath:str,
        scale: int,
        type: str,
        outfilepath:str,
):
    """
            You can use this function if you only want to evaluate one file

            :param csvpath: The CSV file path to be calculated
            :param scale: The scale size of the calculation. The data type is an integer. 
                    One month is 30, three months is 3 times 30, six months is 6 times 60, and so on.
            :param type: str, Calculate the monthly or daily SPI. 
            :param outcsv: The CSV file path to save the result
            :return: no return
    """
    csvfilelist=glob.glob(filepath+'\\'+'*.csv')
    for file in csvfilelist:
        name=os.path.basename(file)
        outfile=outfilepath+'\\SPI'+name
        cal_SPI(file,scale,type,outfile)
    print(file+' has been down!')

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    #calculate a file
    infile=r'C:\Users\Rong\Desktop\50353.csv'
    outfile=r'C:\Users\Rong\Desktop\SPI50353.csv'
    cal_SPI(infile,30,'daily',outfile)

    #calculate multiple files
    # infile = r'C:\Users\Rong\Desktop\for_spi_input'
    # outfile = r'C:\Users\Rong\Desktop\spi_out_scale_30'
    # batch_cal_SPI(infile,30,'daily',outfile)
    print('All down')