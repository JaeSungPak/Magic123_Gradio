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
        epoch=5
        save_mesh_path = "output/Magic123/"
        save_mesh_name = "mesh.glb"
        input_path = "./Magic123_Gradio/input"
        image_path = input_path + "/input.png"

        #Do not modify output_path
        output_path = "./Magic123_Gradio/out"

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
        input_image.save(image_path)

        #run
        cmd = f"python Magic123_Gradio/preprocess_image.py --path {image_path}"
        try:
            completed_process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
            print(completed_process.stdout)
            
            for i in tqdm.tqdm(range(50), desc="Finished image preprocessing..."):
                time.sleep(0.01)
                    
            #Coarse Stage
            main_gradio.run(dmtet=False, iters=epoch*100)
            #Fine Stage
            main_gradio.run(dmtet=True, iters=epoch*100)
            print(completed_process.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.stdout)
            print(e.stderr)

        output_name = f"./Magic123_Gradio/out/magic123-nerf-dmtet/magic123_input_nerf_dmtet/mesh/mesh.glb"
        shutil.copyfile(output_name, f"{save_mesh_path}/mesh.glb")
        
        return f"{save_mesh_path}/mesh.glb"
    
    btn.click(generate_mesh, inputs, outputs)

#image = Image.open("./0.png")
#generate_mesh(image)

#inputs = gr.inputs.Image(label="Image", type="pil")
#outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
#gr.Interface(generate_mesh, inputs, outputs).launch(share=True)

demo.queue(concurrency_count=20).launch(share=True)
