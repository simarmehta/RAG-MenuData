import gradio as gr
import requests

def chatbot_interface(query, history):
     
    url = "http://localhost:8000/query"  
    payload = {"query": query}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}
    
    if "error" in result:
        answer = f"Error: {result['error']}"
        references_text = ""
    else:
        answer = result.get("answer", "")
        internal_refs = result.get("internal_results", [])
        external_refs = result.get("external_results", [])

        references_text = "Internal References:\n"
        for ref in internal_refs:
            
            references_text += f"- {ref.get('sentence', 'N/A')}\n"
        references_text += "\nExternal References:\n"
        for ref in external_refs:
            
            references_text += f"- {ref.get('paragraph', 'N/A')}\n"
    
    if history is None:
        history = []
    history.append((query, answer))
    return history, references_text

def clear_history():
    
    return [], ""


with gr.Blocks() as demo:
    gr.Markdown("# Chatbot Interface")
    
    with gr.Row():
        query_input = gr.Textbox(lines=2, placeholder="Enter your query here.", label="Your Query")
        submit_btn = gr.Button("Submit")
    
  
    chatbot = gr.Chatbot(label="Chat History")

    references_output = gr.Textbox(lines=10, placeholder="References will appear here.", label="References")
    
    clear_btn = gr.Button("Clear Chat")
    
    submit_btn.click(
        fn=chatbot_interface, 
        inputs=[query_input, chatbot], 
        outputs=[chatbot, references_output],
        queue=True
    )
    
    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, references_output]
    )
    
demo.launch()
