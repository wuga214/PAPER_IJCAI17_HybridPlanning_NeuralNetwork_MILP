Data Generation Commands
===

## Reservoir_4
./run rddl.sim.DomainExplorer -R files/Reservoir/Reservoir_4/Reservoir_det.rddl -P rddl.policy.domain.reservoir.StochasticReservoirPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 2000 -D nn/Reservoir/Reservoir_4/Reservoir_Data.txt -L nn/Reservoir/Reservoir_4/Reservoir_Label.txt >log.txt

## Reservoir_8
./run rddl.sim.DomainExplorer -R files/Reservoir/Reservoir_8/Reservoir_det.rddl -P rddl.policy.domain.reservoir.StochasticReservoirPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 2000 -D nn/Reservoir/Reservoir_8/Reservoir_Data.txt -L nn/Reservoir/Reservoir_8/Reservoir_Label.txt >log.txt

## HVAC_6
./run rddl.sim.DomainExplorer -R files/HVAC/ROOM_6/HVAC_VAV.rddl2 -P rddl.policy.domain.HVAC.HVACPolicy_VAV_fix -I inst_hvac_vav_fix -V rddl.viz.ValueVectorDisplay -K 2000 -D nn/HVAC/ROOM_6/HVAC_Data.txt -L nn/HVAC/ROOM_6/HVAC_Label.txt

## HVAC_3
./run rddl.sim.DomainExplorer -R files/HVAC/ROOM_3/HVAC_VAV.rddl2 -P rddl.policy.domain.HVAC.HVACPolicy_VAV_fix -I inst_hvac_vav_fix -V rddl.viz.ValueVectorDisplay -K 2000 -D nn/HVAC/ROOM_3/HVAC_Data.txt -L nn/HVAC/ROOM_3/HVAC_Label.txt

## Navigation_8x8
./run rddl.sim.DomainExplorer -R files/Navigation/8x8/Navigation_Radius.rddl -P rddl.policy.domain.navigation.StochasticNavigationPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 5000 -D nn/Navigation/8x8/Navigation_Data.txt -L nn/Navigation/8x8/Navigation_Label.txt

## Navigation_10x10
./run rddl.sim.DomainExplorer -R files/Navigation/10x10/Navigation_Radius.rddl -P rddl.policy.domain.navigation.StochasticNavigationPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 5000 -D nn/Navigation/10x10/Navigation_Data.txt -L nn/Navigation/10x10/Navigation_Label.txt


Fixed Policy Performance Extraction
===

## Reservoir_4
./run rddl.sim.DomainExplorer -R files/Reservoir/Reservoir_4/Reservoir_det.rddl -P rddl.policy.domain.reservoir.FixReservoirPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/Reservoir_4_Data.txt -L fixedpolicy/Reservoir_4_Label.txt

## Reservoir_8
./run rddl.sim.DomainExplorer -R files/Reservoir/Reservoir_8/Reservoir_det.rddl -P rddl.policy.domain.reservoir.FixReservoirPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/Reservoir_8_Data.txt -L fixedpolicy/Reservoir_8_Label.txt

## HVAC_6
./run rddl.sim.DomainExplorer -R files/HVAC/ROOM_6/HVAC_VAV.rddl2 -P rddl.policy.domain.HVAC.HVACPolicy_VAV_det -I inst_hvac_vav_fix -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/HVAC_6_Data.txt -L fixedpolicy/HVAC_6_Label.txt

## HVAC_3
./run rddl.sim.DomainExplorer -R files/HVAC/ROOM_3/HVAC_VAV.rddl2 -P rddl.policy.domain.HVAC.HVACPolicy_VAV_det -I inst_hvac_vav_fix -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/HVAC_3_Data.txt -L fixedpolicy/HVAC_3_Label.txt

## Navigation_8x8
./run rddl.sim.DomainExplorer -R files/Navigation/8x8/Navigation_Radius.rddl -P rddl.policy.domain.navigation.StochasticNavigationPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/Navigation_8x8_Data.txt -L fixedpolicy/Navigation_8x8_Label.txt

## Navigation_10x10
./run rddl.sim.DomainExplorer -R files/Navigation/10x10/Navigation_Radius.rddl -P rddl.policy.domain.navigation.StochasticNavigationPolicy -I is1 -V rddl.viz.ValueVectorDisplay -K 1 -D fixedpolicy/Navigation_10x10_Data.txt -L fixedpolicy/Navigation_10x10_Label.txt

Bayesian Network Display
===

## Reservoir 4
./run rddl.viz.RDDL2Graph -R files/Reservoir/Reservoir_4/Reservoir_det.rddl -I is1

## Reservoir 8
./run rddl.viz.RDDL2Graph -R files/Reservoir/Reservoir_4/Reservoir_det.rddl -I is1

## HVAC_6
./run rddl.viz.RDDL2Graph -R files/HVAC/ROOM_6/HVAC_VAV.rddl2 -I inst_hvac_vav_fix

## HVAC_3
./run rddl.viz.RDDL2Graph -R files/HVAC/ROOM_3/HVAC_VAV.rddl2 -I inst_hvac_vav_fix

## Navigation_8x8
./run rddl.viz.RDDL2Graph -R files/Navigation/8x8/Navigation_Radius.rddl -I is1

## Navigation_10x10
./run rddl.viz.RDDL2Graph -R files/Navigation/10x10/Navigation_Radius.rddl -I is1




