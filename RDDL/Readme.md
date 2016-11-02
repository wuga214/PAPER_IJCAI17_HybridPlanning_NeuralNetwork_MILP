### Progress
1. Data Extraction from RDDL is functional
2. Domain Reservoir is function

### TODO
1. Policy for Reservoir
2. Domain Description for Inventory control and power gen

### New Helper Class
./run rddl.Help

### New DomainExplorer
./run rddl.sim.DomainExplorer

### New folder to accept data and label output for Neural Network training.

### RUN
Modified run to accept more than 13 parameters

### Command Line Example:

./run rddl.sim.DomainExplorer -R files/final_comp/rddl -P rddl.policy.RandomBoolPolicy -I elevators_inst_mdp__9 -V rddl.viz.ValueVectorDisplay -D nn/Data.txt -L nn/Label.txt