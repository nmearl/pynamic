from pynamic.optimizer import Optimizer
from pynamic.parameters import Parameters
from pynamic.analyzer import Analyzer
from pynamic.watcher import Watcher

# Setup the parameters object by reading in an input file
params = Parameters('test.inp')

# Setup the optimizer, including adding the photometric and rv data
optimizer = Optimizer(params=params,
                      photo_data_file='my_data.txt',
                      rv_data_file='my_rv_data.txt',
                      rv_body=0)

# Start the watcher so we can see what's going on
watcher = Watcher(optimizer)
watcher.start()

# Run the optimizer using the emcee hammer algorithm
optimizer.run(metho='mcmc',
              nprocs=2)

# Stop the watcher
watcher.stop()

# Save out the results for later analysis
optimizer.save('model_data.txt')

# Now let's analyze the results. Pass the optimizer object to the analyzer
analyzer = Analyzer(optimizer)

# Report the results to the screen, and save them
analyzer.report()

# Now we can make some histrograms
analyzer.save('flux')