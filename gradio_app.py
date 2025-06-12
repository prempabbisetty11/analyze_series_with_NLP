import gradio as gr
import pandas as pd
from theme_classifier import ThemeClassifier
from character_network import NameEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
import os
from dotenv import load_dotenv
load_dotenv()
from character_chatbot import CharacterChatbot


def get_themes(theme_list_str, subtitles_list, save_path):
    # Lowercase and strip user input
    theme_list = [theme.strip().lower() for theme in theme_list_str.split(',') if theme.strip() and theme.strip().lower() != 'dialogue']
    
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_list, save_path)

    # Lowercase and strip DataFrame columns
    output_columns = [col.strip().lower() for col in output_df.columns]

    print("\n Output DataFrame Columns:", list(output_df.columns))
    print(" User Provided Themes:", theme_list)

    # Find valid themes by matching lowercased/stripped names
    valid_themes = [output_df.columns[i] for i, col in enumerate(output_columns) if col in theme_list]

    if not valid_themes:
        print(" None of the themes you entered matched the model output.")
        print("Available output columns were:", output_df.columns.tolist())
        return pd.DataFrame({
            "Theme": ["No matching themes found. Available columns: " + ", ".join(output_df.columns)],
            "Score": [0]
        })

    theme_scores = output_df[valid_themes].sum().reset_index()
    theme_scores.columns = ['Theme', 'Score']
    return theme_scores

def get_character_network(subtitles_path, ner_path):
    ner = NameEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html
 
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    justu_classifier = JutsuClassifier(
        model_path=text_classification_model,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv('huggingface_token')
    )
    output = justu_classifier.classify_jutsu(text_to_classify)
    output = output[0]
    return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatbot("AbdullahTarek/Naruto_Llama-3-8B_3",
                                         huggingface_token=os.getenv('huggingface_token')
                                         )
    
    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output

def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot(
                            x='Theme',
                            y='Score',
                            title="Series Themes",
                            tooltip=["Theme", "Score"],
                            vertical=False,
                            width=500,
                            height=260
                        )
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path_1 = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path_1, save_path], outputs=[plot])

        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path_2 = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NERs Save Path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path_2, ner_path], outputs=[network_html])

        # Text Classification with LLms
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLms</h1>")
                with gr.Row():
                    with gr.Column():
                        test_classification_output = gr.Textbox(label="Test Classification Output")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model Path")
                        text_classification_data_path = gr.Textbox(label="Data Path")
                        text_to_classify = gr.Textbox(label="Text Input")
                        classify_text_button = gr.Button("Classify Text (Justu)")
                        classify_text_button.click(
                            classify_text,
                            inputs=[text_classification_model, text_classification_data_path, text_to_classify],
                            outputs=[test_classification_output]
                        )
        
        # Character Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)

    

    iface.launch(share=True)

if __name__ == '__main__':
    main()