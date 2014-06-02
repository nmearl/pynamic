from pynamic.optimizer import Optimizer
from pynamic.parameters import Parameters
from pynamic.analyzer import Analyzer
from pynamic.watcher import Watcher

# Setup the parameters object by reading in an input file
params = Parameters('')

# Setup the optimizer, including adding the photometric and rv data
optimizer = Optimizer(params=params,
                      photo_data_file='',
                      rv_data_file='',
                      rv_body=1)

# Start the watcher so we can see what's going on
watcher = Watcher(optimizer)
watcher.start()

# Run the optimizer using the emcee hammer algorithm
optimizer.run('mcmc', nprocs=2, niterations=5)

# Stop the watcher
watcher.stop()

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