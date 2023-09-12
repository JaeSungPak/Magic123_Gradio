import os
import activation
import gradio as gr
import subprocess
from PIL import Image
import numpy as np
import shutil
import time
import tqdm
import main_gradio

with gr.Blocks() as demo:
    inputs = gr.inputs.Image(label="Image", type="pil")
    outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
    btn = gr.Button("Generate!")
    
    def generate_mesh(input_image, progress=gr.Progress(track_tqdm=True)):

        #Modify epoch or save_mesh_path as needed!
        epoch=1
        save_mesh_path = "output/Magic123/"
        save_mesh_name = "mesh.glb"

        #Do not modify output_path
        output_path = "./Magic123_Gradio/out"
        input_path = "./Magic123_Gradio/input"
        image_name = "input.png"

        #Create the folders needed for processing
        if os.path.exists(input_path):
            shutil.rmtree(input_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        if os.path.exists(save_mesh_path):
            shutil.rmtree(save_mesh_path)
        if os.path.exists("output") == False:
            os.mkdir("output")

        os.mkdir(input_path)
        os.mkdir(save_mesh_path)
        input_image.save(f"{input_path}/{image_name}")

        #run
        cmd = f"python Magic123_Gradio/preprocess_image.py --path {input_path}/{image_name}"
        try:
            completed_process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
            
            for i in tqdm.tqdm(range(50), desc="Finished image preprocessing..."):
                time.sleep(0.01)
                    
            main_gradio.generate_mesh(epoch)
            
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.stdout)
            print(e.stderr)

        output_name = f"./Magic123_Gradio/out/magic123-nerf-dmtet/magic123_input_nerf_dmtet/mesh/mesh.glb"
        shutil.copyfile(output_name, f"{save_mesh_path}/{save_mesh_name}")
        
        return f"{save_mesh_path}/{save_mesh_name}"
    
    btn.click(generate_mesh, inputs, outputs)

#image = Image.open("./0.png")
#generate_mesh(image)

#inputs = gr.inputs.Image(label="Image", type="pil")
#outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
#gr.Interface(generate_mesh, inputs, outputs).launch(share=True)

demo.queue().launch(share=True)
