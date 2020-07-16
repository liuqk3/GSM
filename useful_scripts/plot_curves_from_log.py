
import matplotlib.pyplot as plt
import os

def get_data(log_file, key='loss', phase='train'):
    """This function get the data from log file

    Args:
        log_file: str, the path of log file
        key: str, the key word to get the loss
    """

    values = []
    itrs = []

    record = ''
    start_record = False
    with open(log_file) as log_file:
        for line in log_file:

            if '[Train' in line:
                start_record = True
            if not start_record:
                continue

            if '[Train' in line:
                phase_cur = 'train'
                new_record = True
            else:
                if phase_cur == 'train':
                    new_record = False

            if '[Valid' in line:
                phase_cur = 'valid'
                new_record = True
            else:
                if phase_cur == 'valid':
                    new_record = False

            if phase_cur != phase:
                continue

            if new_record: # a new record
                # handle the record
                record = record.replace('\t', '')
                record = record.replace('\n', '')
                record = record.strip().split(',')
                # get key value
                value = None
                for r in record:
                    if key + ':' in r:
                        value = r
                        break
                if value is None:
                    record = line
                    continue
                value = float(value.split(':')[-1])
                
                # get iter
                epoch = record[0].split('Epoch')[-1] # such as 0/30
                epoch = int(epoch.split('/')[0])
                
                itr = record[0].split(':')[-1]  # such as 10/1023
                itr = itr.split('/')
                itr = int(itr[0]) + epoch * int(itr[1])
                
                values.append(value)
                itrs.append(itr)

                # start a new record
                record = line
            else:
                record = record + ', ' + line

    return itrs, values


def plot_curv(x, y, name='', save_path=None):
    plt.figure()
    plt.plot(x, y)
    plt.title(name)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        print('figure saved in {}'.format(save_path))


if __name__ == '__main__':

    key = 'loss'
    log_file = 'logs/GraphSimilarity_v5/9/logs.txt'
    itrs, values = get_data(log_file=log_file, key=key)

    # save path
    save_path = log_file.replace('logs.txt', key+'.png')

    plot_curv(x=itrs, y=values, name=key, save_path=save_path)
    plt.show()

    a = 1




