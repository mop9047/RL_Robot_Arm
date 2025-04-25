import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("armModel_threedof/threedof.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 🛠 Set control values using sliders in the GUI
        viewer.sync()
        
        # 🚀 Apply the control signals from sliders to the model
        mujoco.mj_step(model, data)

