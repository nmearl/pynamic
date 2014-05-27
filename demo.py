__author__ = 'nmearl'

from pynamic.optimizer import Optimizer
from pynamic.parameters import Parameters
from pynamic.analyzer import Analyzer
from pynamic.watcher import Watcher
import time

# Setup the parameters object
params = Parameters('input/koi126.inp')

# Setup the optimizer, including adding the photometric and rv data
optimizer = Optimizer(params=params,
                      photo_data_file='data/005897826_part_llc.txt',
                      rv_data_file='data/005897826_rv.dat',
                      rv_body=3)

now = time.time()
# Start the watcher so we can see what's going on
watcher = Watcher(optimizer)
watcher.start()

# Run the optimizer using the emcee hammer algorithm
optimizer.run('SLSQP')  #, nprocs=2, niterations=5)

# Stop the watcher
watcher.stop()

print("Time: {0}".format(time.time() - now))

# Save out the results for later analysis
optimizer.save()

# Now let's analyze the results. Pass the optimizer object to the analyzer
analyzer = Analyzer(optimizer)

# Report the results to the screen, and save them
analyzer.report()

# Now we can make some histrograms
# analyzer.save('histogram')
# analyzer.save('step')
analyzer.show('flux')
