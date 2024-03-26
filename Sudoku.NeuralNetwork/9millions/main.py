from loguru import logger
from train import train


if __name__ == '__main__':
    logger.debug("start main")
    train(r'D:\epita\ubuntu\SCIA-1\programmationParContrainte\datasets_sudoku\sudoku.csv')