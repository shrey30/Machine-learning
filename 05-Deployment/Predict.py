# # Load the Model
import pickle

model_file = 'model_C=1.bin'

with open(model_file, 'rb') as f_in: # 'rb' because we are reading into binary file.
    dv, model = pickle.load(f_in)
