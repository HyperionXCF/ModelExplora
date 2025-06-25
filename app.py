import gradio as gr
import os
from huggingface_hub import InferenceClient
from functools import partial
from dotenv import load_dotenv
load_dotenv()
import random 

def random_prompt():
    return random.choice([
        "Explain Inheritence in Object Oriented Programming",
        "Explain Bubble Sort Algorithm with C++ code",
        "how to write a lambda function in python to calculate square of a number",
        "Tell me a fun fact about programming",
        "What is the next big thing in AI / ML ?",
        "What could be possible outcome of Human AI ?"
    ])

system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

custom_css = """
body{
    /* WebKit-based browsers (Chrome, Safari, etc.) */
.element-with-scroll::-webkit-scrollbar {
    display: none;
}
/*  Alternatively, to hide the scrollbar and prevent scrolling */
.element-with-scroll {
    overflow: hidden;
}
}
.gradio-container {
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');
    font-family:"Lexend", serif;
    max-width: 700px; /* Adjust this value as needed (e.g., 800px, 1200px) */
    margin: auto;    /* Centers the container horizontally */
}
/* You might also want to ensure responsiveness for smaller screens */
@media (max-width: 768px) { /* Example for smaller screens */
    .gradio-container {
        max-width: 95%; /* Use a percentage for better responsiveness */
        margin: auto;
    }
}
"""

def inference(prompt, hf_token, model, model_name):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HF_TOKEN")
    client = InferenceClient(model=model, token=hf_token)
    tokens = f"**`{model_name}`**\n\n"
    for completion in client.chat_completion(messages, max_tokens=300, stream=True):
        token = completion.choices[0].delta.content
        tokens += token
        yield tokens

def hide_textbox():
    return gr.Textbox(visible=False)

with gr.Blocks(theme=gr.themes.Ocean(),css=custom_css,title="Open Model Explorer") as demo:
    gr.Markdown("<center><h2>HF Model Explorer</h2></center>")
    gr.Markdown("<br>")    
    gr.Markdown("<center><h6></h6></center>")
    gr.Markdown("<h6>Every LLM has its own personality! type your prompt below and compare results to choose the best LLM for your next project !</h6>")
    gr.Markdown("A small hobby project to learn Gradio and HuggingFace inference API, inspired by Google's AI studio.My goal is to expand this functionality so users can input any model name from the Hugging Face library to test them out.")
    gr.Markdown("<br>")
    
    prompt = gr.Textbox(value=random_prompt,label="Prompt",lines=3 )
    token = gr.Textbox(label="Enter your own Hugging Face Token if error is shown (otherwise keep empty)", type="password", placeholder="Enter your HuggingFace API token")
    #by default gradio arranges everything in a column 
    #for putting things in a row use : gr.Row()
    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button(variant="primary") 
            code_btn = gr.Button("View Code",variant="secondary") 
    
    with gr.Row():
        llama_output = gr.Markdown("<center> Output from LLama 3-70B instruct</center>")
        nous_output = gr.Markdown("<center>Output from Nous Hermes 2 Mixtral </center>")
        
        
    # generate_btn.click(
    #     fn = add,
    #     inputs = [prompt,token],
    #     outputs = [llama_output,zephyr,nous_output],
    #     show_progress = "hidden"
    # )
    
    # prompt.submit(
    #     fn = add,
    #     inputs = [prompt,token],
    #     outputs = [llama_output,zephyr,nous_output],
    #     show_progress = "hidden"
    # )
    
    # #instead of creating 2 different event handlers use gr.on()
    # gr.on(
    #     triggers = [prompt.submit, generate_btn.click],
    #     fn = hide_textbox,
    #     inputs = None,
    #     outputs = [token]
    #     # after clicking the run button we want to hide the token
    #     # therefore we can create multiple functions and call them one after another when the event happens. will create hide box fn 
    # )
    
    gr.on(
        triggers = [prompt.submit,generate_btn.click],
        fn = partial(inference,model = "meta-llama/Meta-Llama-3-70b-Instruct",model_name="Llama 3-70b Instruct"),
        inputs = [prompt,token],
        outputs = [llama_output],
        show_progress = "hidden" 
    )
    gr.on(
        triggers = [prompt.submit,generate_btn.click],
        fn = partial(inference, model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", model_name="Nous Hermes 2 Mixtral 8x7B DPO"),
        inputs = [prompt,token],
        outputs = [nous_output],
        show_progress = "hidden" 
    )
    gr.Markdown("<br>")
    
demo.launch(share=True)