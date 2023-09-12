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
                
        input_path = "./Magic123_Gradio/input"
        output_path = "./Magic123_Gradio/out"
        image_path = input_path + "/input.png"
        save_mesh_path = "output/Magic123/"
        GPU_NUM = "0"

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

        cmd = f"python Magic123_Gradio/preprocess_image.py --path {image_path}"        #cmd_2 = f"bash scripts/magic123/run_both_priors.sh {GPU_NUM} nerf dmtet {input_path} 1 1"
        try:
            completed_process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
            print(completed_process.stdout)
            
            for i in tqdm.tqdm(range(50), desc="Finished image preprocessing..."):
                time.sleep(0.01)
                    
            #completed_process = subprocess.run(cmd_2.split(), stdout=subprocess.PIPE)
            main_gradio.run(False, save_mesh_path=save_mesh_path,iters=100)
            main_gradio.run(True, save_mesh_path=save_mesh_path,iters=100)
            print(completed_process.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            print(e.stdout)
            print(e.stderr)

        output_name = f"Magic123_Gradio/out/magic123-nerf-dmtet/magic123_input_nerf_dmtet/mesh/mesh.glb"
        shutil.copytree(output_name, save_mesh_path)
        
        return save_mesh_path
    
    btn.click(generate_mesh, inputs, outputs)

#image = Image.open("./0.png")
#generate_mesh(image)

#inputs = gr.inputs.Image(label="Image", type="pil")
#outputs = gr.Model3D(label="3D Mesh", clear_color=[1.0, 1.0, 1.0, 1.0])
#gr.Interface(generate_mesh, inputs, outputs).launch(share=True)

demo.queue(concurrency_count=20).launch(share=True)
