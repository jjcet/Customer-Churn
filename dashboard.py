from zenml.environment import Environment
from pyngrok import ngrok

def start_zenml_dashboard(port=8237):
    public_url = ngrok.connect(port)
    print(f"\x1b[31mIn Colab, use this URL instead : {public_url}!\x1b[0m]")
    !zenml up --blocking --port {port}


start_zenml_dashboard()