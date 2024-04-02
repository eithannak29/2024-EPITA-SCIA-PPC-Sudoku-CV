from loguru import logger
from train import train


def test(message):
    s = "let's go" + message
    return s


if __name__ == '__main__':
    logger.debug("start main")
    print("alors peut etre")
    train(r'D:\epita\ubuntu\SCIA-1\programmationParContrainte\datasets_sudoku\sudoku.csv')
    print("ouioui")