import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
from dateutil.tz import gettz
import sys

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)
import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath
from twin4build.api.models.OE20_601b_2_model import fcn
    
def get_model():
    model = tb.Model(id="model_OE20_601b_2", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_OU44_room_case_renamed.xlsm")
    # model.load_model(semantic_model_filename=filename, infer_connections=True, create_signature_graphs=False, validate_model=False)
    model.load_model(fcn=fcn, infer_connections=False)
    return model


def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()
    monitor = tb.Monitor(model)
    monitor.monitor(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    show=False)
    
    plot.plot_outdoor_environment(model, monitor.simulator, firstAxisylim=[5, 30])
    space_name = "[020B][020B_space_heater]"


    #########################################
    print(model.component_dict["[020B][020B_space_heater]"].savedOutput["spaceHeaterPower"])
    #########################################


    temperature_heating_controller_name = "Temperature controller"
    plot.plot_space_temperature_fmu(model, monitor.simulator, space_name)
    plot.plot_space_CO2_fmu(model, monitor.simulator, space_name)
    plot.plot_temperature_controller(model, monitor.simulator, temperature_heating_controller_name, show=True)

if __name__ == "__main__":
    run()