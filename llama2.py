import gradio as gr
from vllm import LLM, SamplingParams
system_message = "Answer prompt questions Briefly"
prompt_template=f'''Below is an instruction that describes a task. Write a response that adequately completes the request.
### Instruction:
{{prompt}}

### Answer:
'''

llm = LLM(
    model="TheBloke/Llama-2-7B-ft-instruct-es-AWQ",
    quantization="awq",
    max_model_len=500,
    gpu_memory_utilization=0.7,
    dtype="half"
)
def generate_text(prompt,max_tokens=50):
    prompt_with_template = prompt_template.format(prompt=prompt)
    outputs = llm.generate([prompt_with_template], (SamplingParams(temperature=0, max_tokens=max_tokens)))
    generated_text = outputs[0].outputs[0].text if outputs else ""
    return generated_text
iface = gr.Interface(
    fn=generate_text,
    inputs=["text", gr.Slider(1,200,1, label="Max Tokens")],
    outputs="text",
    title="TheBloke/Llama-2-7B-ft-instruct-es-AWQ",
    description="Enter a prompt to generate text.",
)
iface.launch()
