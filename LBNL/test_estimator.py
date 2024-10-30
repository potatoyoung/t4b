import os
import datetime
import json
import sys
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 3), "BuildingEnergyModel")
    sys.path.append(file_path)
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from test_LBNL_model import extend_model
def test_estimator():
    stepSize = 60
    # startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
    # endPeriod = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0)

    model = Model(id="LBNL", saveSimulationResult=True)
    model.load_model(infer_connections=False, extend_model=extend_model)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    # Dictinary inputs for the estimation/optimization method.
    # Each dictionary follow the following convention:
    # {object: "attribute name"}


# array([ 7.04131076e-01,  5.00000000e+00,  1.53704878e+01,  1.04948179e+01,
#         1.92074356e+01,  2.57586300e+03,  1.21013000e+05,  6.57547285e-01,
#         2.98661582e+00,  3.14267201e-02,  3.94328405e-01, -7.03136971e-02,
#         4.04592247e-01])


    # x0 = {coil: [3.591852189575182, 5, 18.41226242542397, 10.64284456709903, 13.976768318598277, 2568.0665571292134, 121013.0],
    #     valve: [0.7675164465562464, 3.171332962392209],
    #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
    
    # x0 = {coil: [1, 5, 18.41226242542397, 10.64284456709903, 13.976768318598277, 1000, 96000],
    #     valve: [0.8, 1],
    #     fan: [0.01938953650131158, 0.4266115043400285, -0.061931356081560654, 0.4148001183354297]}
    
    # lb = {coil: [0.5, 0.5, 1, 1, 1, 0, 1000],
    #     valve: [0.5, 0],
    #     fan: [-1, -1, -1, -1]}
    
    # ub = {coil: [10, 10, 50, 50, 50, 10000, 300000],
    #     valve: [1, 5],
    #     fan: [2, 2, 2, 2]}
    
    # targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue", "nominalSensibleCapacity.hasValue"],
    #                         valve: ["valveAuthority", "waterFlowRateMax"],
    #                         fan: ["c1", "c2", "c3", "c4"]}
    # targetMeasuringDevices = [model.component_dict["coil outlet air temperature sensor"],
    #                             model.component_dict["coil outlet water temperature sensor"],
    #                             model.component_dict["fan power meter"]]
    
    
    # endPeriod = datetime.datetime(year=2022, month=2, day=2, hour=23, minute=0, second=0)


    n_days = 2
    startPeriod = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0)

    startPeriod_train = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0)
    endPeriod_train = datetime.datetime(year=2022, month=2, day=1, hour=21, minute=0, second=0)
    startPeriod_test = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0)
    endPeriod_test = datetime.datetime(year=2022, month=2, day=15, hour=0, minute=0, second=0)
    sol_list = []
    
    endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]
    startPeriod_list = [startPeriod]*len(endPeriod_list)

    # endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]
    # startPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(0, n_days-1, 2)]

    n_estimations = 1
    startPeriod_list = [startPeriod for i in range(n_estimations)]
    endPeriod_list = [startPeriod + datetime.timedelta(days=1) for i in range(n_estimations)]
    
    overwrite = True
    filename = 'DryCoilDiscretized_test_fmu_valve_controller.json'
    # filename = 'DryCoilEffectivenessNTU.json'
    # filename = 'test.json'

    if overwrite==False and os.path.isfile(filename):
        with open(filename, 'r') as f:
            sol_dict = json.load(f)
    else:
        sol_dict = {}
        
    print(sol_dict.keys())
    for i, (startPeriod, endPeriod) in enumerate(zip(startPeriod_list, endPeriod_list)):
        if str(i) not in sol_dict.keys():
            print("-------------------------------")
            print(i)

            startPeriod = startPeriod_train #######################################
            endPeriod = endPeriod_train #######################################

            # x0 = {coil: [1.5, 10, 300, 20, 50, 8000],
            #     valve: [5000, 10, 3],
            #     fan: [0.0015302446, 0.0052080574, 1.1086242, -0.11635563, 0.9, 0.8],
            #     controller: [5, 5, 5]}
            
            # lb = {coil: [0.5, 3, 1, 1, 1, 500],
            #     valve: [1000, 1, 0.5],
            #     fan: [-1, -1, -1, -1, 0.7, 0],
            #     controller: [0, 0, 0]}
            
            # ub = {coil: [5, 15, 30, 30, 30, 30000],
            #     valve: [5e+6, 200, 5],
            #     fan: [1.5, 1.5, 1.5, 1.5, 1, 1],
            #     controller: [100, 100, 100]}


            # targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
            #                         valve: ["workingPressure.hasValue", "flowCoefficient.hasValue", "waterFlowRateMax"],
            #                         fan: ["c1", "c2", "c3", "c4", "eps_motor", "f_motorToAir"],
            #                         controller: ["kp", "Ti", "Td"]}


            ####################################### FEWER PARAMETERS ############################################################
            # x0 = {coil: [1.5, 10, 15, 15, 15, 8000],
            #     valve: [100000, 1.5],
            #     fan: [0.027828, 0.026583, -0.087069, 1.030920, 1],
            #     controller: [50, 50, 50]}
            
            # lb = {coil: [0.5, 3, 1, 1, 1, 500],
            #     valve: [1000, 0.5],
            #     fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
            #     controller: [0, 0, 0]}
            
            # ub = {coil: [5, 15, 30, 30, 30, 30000],
            #     valve: [1e+6, 5],
            #     fan: [0.2, 1.4, 1.4, 1.4, 1],
            #     controller: [100, 100, 100]}


            # targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
            #                         valve: ["dpFixed_nominal", "waterFlowRateMax"],
            #                         fan: ["c1", "c2", "c3", "c4", "f_total"],
            #                         controller: ["kp", "Ti", "Td"]}
            #################################################################################################################

            ####################################### VALVE BYPASS ############################################################
            x0 = {coil: [1.5, 10, 15, 15, 15, 1500],
                valve: [1.5, 1.5, 10000, 2000, 1e+6, 1e+6, 5],
                fan: [0.027828, 0.026583, -0.087069, 1.030920, 0.9],
                controller: [50, 50, 50]}
            
            lb = {coil: [0.5, 3, 1, 1, 1, 500],
                valve: [0.5, 0.5, 100, 100, 100, 100, 0.1],
                fan: [-0.2, -0.7, -0.7, -0.7, 0.7],
                controller: [0.05, 1, 0]}
            
            ub = {coil: [5, 15, 30, 30, 30, 3000],
                valve: [2, 5, 1e+5, 1e+5, 5e+6, 5e+6, 500],
                fan: [0.2, 1.4, 1.4, 1.4, 1],
                controller: [100, 100, 100]}


            targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
                                    valve: ["mFlowValve_nominal", "mFlowPump_nominal", "dpCheckValve_nominal", "dpCoil_nominal", "dpPump", "dpSystem", "riseTime"],
                                    fan: ["c1", "c2", "c3", "c4", "f_total"],
                                    controller: ["kp", "Ti", "Td"]}
            #################################################################################################################
            
            percentile = 2
            targetMeasuringDevices = {model.component_dict["coil outlet air temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                        model.component_dict["coil outlet water temperature sensor"]: {"standardDeviation": 0.5/percentile},
                                        model.component_dict["fan power meter"]: {"standardDeviation": 80/percentile},
                                        model.component_dict["valve position sensor"]: {"standardDeviation": 0.01/percentile}}
            

            options = {"n_sample": 1, 
                    "n_temperature": 1, 
                    "fac_walker": 2,
                    "prior": "uniform",
                    "walker_initialization": "uniform"}
        
            estimator.estimate(x0=x0,
                                lb=lb,
                                ub=ub,
                                targetParameters=targetParameters,
                                targetMeasuringDevices=targetMeasuringDevices,
                                startPeriod=startPeriod_train,
                                endPeriod=endPeriod_train,
                                startPeriod_test=startPeriod_test,
                                endPeriod_test=endPeriod_test,
                                stepSize=stepSize,
                                method="MCMC",
                                options=options
                                )
            
if __name__=="__main__":
    test_estimator()