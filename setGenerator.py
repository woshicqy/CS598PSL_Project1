from email import header
import numpy as np
import pandas as pd

data = pd.read_csv('Ames_data.csv')
testID = pd.read_csv('project1_testIDs.dat',sep=' ')
data_top = list(data.columns.values)
# print(f'data top:{data_top}')
# exit()


num_sets = 10
data_arr = data.to_numpy()
testID_arr = testID.to_numpy()

print(f'testID:{testID_arr[:]}')
print(f'testID:{testID_arr[:].shape}')


for i in range(num_sets):
    train_tmp = data_arr[-testID_arr[:,i], :]
    test_tmp = data_arr[testID_arr[:,i], ]
    test_tmp_y = test_tmp[:,-1]
    test_tmp = test_tmp[:,:-1]
    # print(f'train_tmp:{test_tmp}')

    # print(f'test_tmp_y:{test_tmp_y.shape}')
    train_file = 'train' + str(i+1) + '.csv'
    test_file = 'test' + str(i+1) + '.csv'
    test_y_file = 'test_y' + str(i+1) + '.csv'
    np.savetxt(train_file, train_tmp, delimiter=",",fmt='%s',header=','.join(data_top),comments='')
    np.savetxt(test_file, test_tmp, delimiter=",",fmt='%s',header=','.join(data_top))
    np.savetxt(test_y_file, test_tmp_y, delimiter=",")
    print('%d set saving is done!'%(i+1))

