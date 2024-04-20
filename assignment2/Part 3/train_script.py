import train
import numpy as np

train_loss, train_acc, test_loss, test_acc = train.main()

data_to_save = np.column_stack((train_acc, train_loss, test_acc, test_loss))
np.savetxt('output_t=5.txt', data_to_save, header='train_acc train_loss test_acc test_loss')

print("Train Complete!")