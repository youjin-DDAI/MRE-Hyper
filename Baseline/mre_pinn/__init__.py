import os
os.environ['DDEBACKEND'] = 'pytorch'
import deepxde

from . import (
	data,
	training,
	testing,
	baseline,
	fields,
	pde,
	visual,
	utils
)
