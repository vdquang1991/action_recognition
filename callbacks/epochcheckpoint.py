# import the necessary packages
from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0, multi_GPU=False, save_weight_only=False):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt
		self.multi_GPU = multi_GPU
		self.save_weight_only = save_weight_only

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch + 1) % self.every == 0:
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.hdf5".format(self.intEpoch + 1)])
			if self.multi_GPU:
				save_model = self.model.layers[-2]
			else:
				save_model = self.model
			save_model.save(p, overwrite=True)
			if self.save_weight_only:
				p = os.path.sep.join([self.outputPath,
									  "epoch_{}.h5".format(self.intEpoch + 1)])
				save_model.save_weights(p, overwrite=True)
		# increment the internal epoch counter
		self.intEpoch += 1